"""Bidirectional temporal memory on top of the P23 full-frame backbone."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.temporal_frame_net import TemporalFrameNet, _group_count


def _interpolate(input_tensor, **kwargs):
    """Use deterministic CPU interpolation for strict paired runs."""
    if (
        input_tensor.is_cuda
        and os.environ.get('EVSOD_DETERMINISTIC_WARP_CPU', '').strip() == '1'
    ):
        return F.interpolate(input_tensor.cpu(), **kwargs).to(input_tensor.device)
    return F.interpolate(input_tensor, **kwargs)


class TemporalSelfAttentionMemory(nn.Module):
    """Temporal self-attention memory module.

    Core ideas:
    - Self-attention along the time dimension for every spatial position
    - Any two frames can interact in one step, no recurrent propagation needed
    - Spatial pooling to save memory
    - A shared relative-time bias can distinguish nearby, distant, past, and
      future evidence without tying the model to an absolute video length.
    - Configurable small residual initialization preserves the original predictions
      while allowing a non-zero projection to train the attention branch immediately.
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        pool_size=16,
        output_init_std=0.0,
        relative_bias_enabled=False,
        relative_bias_max_distance=8,
    ):
        super().__init__()
        output_init_std = float(output_init_std)
        if output_init_std < 0.0:
            raise ValueError('output_init_std must be non-negative.')
        relative_bias_max_distance = int(relative_bias_max_distance)
        if relative_bias_enabled and relative_bias_max_distance <= 0:
            raise ValueError('relative_bias_max_distance must be positive.')
        self.pool_size = pool_size
        self.relative_bias_enabled = bool(relative_bias_enabled)
        self.relative_bias_max_distance = relative_bias_max_distance
        self.attention = nn.MultiheadAttention(
            channels,
            num_heads,
            batch_first=True,
        )
        self.output_projection = nn.Linear(channels, channels)
        self.relative_position_bias = None
        if self.relative_bias_enabled:
            self.relative_position_bias = nn.Parameter(
                torch.zeros(relative_bias_max_distance * 2 + 1)
            )

        if output_init_std == 0.0:
            nn.init.zeros_(self.output_projection.weight)
        else:
            nn.init.normal_(
                self.output_projection.weight,
                mean=0.0,
                std=output_init_std,
            )
        nn.init.zeros_(self.output_projection.bias)

    def _relative_attention_bias(self, sequence_length, device, dtype):
        if not self.relative_bias_enabled:
            return None
        positions = torch.arange(sequence_length, device=device)
        offsets = positions.unsqueeze(0) - positions.unsqueeze(1)
        indices = offsets.clamp(
            -self.relative_bias_max_distance,
            self.relative_bias_max_distance,
        ) + self.relative_bias_max_distance
        return self.relative_position_bias[indices].to(dtype=dtype)

    def forward(self, bottlenecks):
        """Args:
            bottlenecks: Temporal feature sequence [B, T, C, H, W]

        Returns:
            torch.Tensor: Residual features [B, T, C, H, W]
        """
        B, T, C, H, W = bottlenecks.shape

        pooled = bottlenecks
        h, w = H, W
        if self.pool_size and (H > self.pool_size or W > self.pool_size):
            flat = pooled.reshape(B * T, C, H, W)
            flat = F.adaptive_avg_pool2d(flat, (self.pool_size, self.pool_size))
            h, w = self.pool_size, self.pool_size
            pooled = flat.reshape(B, T, C, h, w)

        tokens = pooled.permute(0, 3, 4, 1, 2).reshape(B * h * w, T, C)

        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
            attn_mask=self._relative_attention_bias(T, tokens.device, tokens.dtype),
        )

        residual = self.output_projection(attended)
        residual = residual.reshape(B, h, w, T, C).permute(0, 3, 4, 1, 2)

        if (h, w) != (H, W):
            flat_r = residual.reshape(B * T, C, h, w)
            flat_r = _interpolate(
                flat_r, size=(H, W), mode='bilinear', align_corners=False,
            )
            residual = flat_r.reshape(B, T, C, H, W)

        return residual


def advect_warp(features, flow, exact_identity=False):
    """Warp features with a dense displacement field.

    The sampling grid is the identity grid plus ``flow``. Therefore a zero
    flow field is an exact identity transform, which preserves a released
    temporal-memory model when the advection branch is first attached.
    """
    if features.ndim != 4:
        raise ValueError('features must have shape [B, C, H, W].')
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError('flow must have shape [B, 2, H, W].')
    if flow.shape[0] != features.shape[0] or flow.shape[2:] != features.shape[2:]:
        raise ValueError('flow shape must match feature batch and spatial dimensions.')
    batch_size, _, height, width = features.shape
    if height <= 1 or width <= 1:
        raise ValueError('features must be larger than one pixel in both dimensions.')

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=features.device, dtype=features.dtype),
        torch.linspace(-1.0, 1.0, width, device=features.device, dtype=features.dtype),
    )
    base_grid = torch.stack((grid_x, grid_y), dim=-1)
    base_grid = base_grid.unsqueeze(0).expand(batch_size, height, width, 2)
    displacement = torch.stack(
        (
            flow[:, 0] * (2.0 / (width - 1)),
            flow[:, 1] * (2.0 / (height - 1)),
        ),
        dim=-1,
    )
    if (
        features.is_cuda
        and os.environ.get('EVSOD_DETERMINISTIC_WARP_CPU', '').strip() == '1'
    ):
        # PyTorch 1.9 has no deterministic CUDA grid-sampler backward.  The
        # M134 pair audit opts into this CPU fallback at the bottleneck scale;
        # device copies preserve autograd while retaining the exact bilinear
        # warp semantics used by the normal path.
        warped = F.grid_sample(
            features.cpu(),
            (base_grid + displacement).cpu(),
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        ).to(features.device)
    else:
        warped = F.grid_sample(
            features,
            base_grid + displacement,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
    if exact_identity:
        # grid_sample at an identity grid has tiny interpolation roundoff in
        # PyTorch 1.9. Keep the exact forward value while retaining the warp
        # gradient needed by the first advection-consistency update.
        return features.detach() + (warped - warped.detach())
    return warped


def advection_consistency_loss(
    features_prev,
    features_curr,
    flow,
    exact_identity=False,
):
    """Align adjacent bottlenecks without requiring motion annotations."""
    return F.mse_loss(
        advect_warp(
            features_prev,
            flow,
            exact_identity=exact_identity,
        ),
        features_curr.detach(),
    )


class FlowEstimationHead(nn.Module):
    """Estimate bottleneck-space displacement between adjacent time bins."""

    def __init__(
        self,
        channels,
        hidden_channels=None,
        max_displacement=0.0,
        normalization_max_groups=8,
    ):
        super().__init__()
        channels = int(channels)
        hidden = int(hidden_channels) if hidden_channels else max(channels // 4, 8)
        max_displacement = float(max_displacement)
        normalization_max_groups = int(normalization_max_groups)
        if max_displacement < 0.0:
            raise ValueError('max_displacement must be non-negative.')
        if normalization_max_groups <= 0:
            raise ValueError('normalization_max_groups must be positive.')
        self.max_displacement = max_displacement
        self.first = nn.Sequential(
            nn.Conv2d(channels * 2, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(
                _group_count(
                    hidden,
                    maximum_groups=normalization_max_groups,
                ),
                hidden,
            ),
            nn.ReLU(inplace=True),
        )
        self.second = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Conv2d(hidden, 2, 3, padding=1, bias=True)
        nn.init.zeros_(self.last.weight)
        nn.init.zeros_(self.last.bias)

    def forward(self, previous, current):
        if previous.shape != current.shape:
            raise ValueError('Adjacent bottleneck features must have the same shape.')
        raw_flow = self.last(
            self.second(self.first(torch.cat((previous, current), dim=1)))
        )
        if self.max_displacement > 0.0:
            return self.max_displacement * torch.tanh(raw_flow)
        return raw_flow

    def has_zero_output(self):
        """Whether the zero-initialized output layer is still unchanged."""
        return not bool(
            torch.count_nonzero(self.last.weight.detach()).item()
            or torch.count_nonzero(self.last.bias.detach()).item()
        )


class ConvGRUCell(nn.Module):
    """A compact spatial ConvGRU cell used only at the U-Net bottleneck."""

    def __init__(self, channels):
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError('channels must be positive.')
        self.channels = channels
        self.gates = nn.Conv2d(channels * 2, channels * 2, 3, padding=1)
        self.candidate = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(self, inputs, state=None):
        if inputs.ndim != 4:
            raise ValueError('inputs must have shape [B, C, H, W].')
        if inputs.shape[1] != self.channels:
            raise ValueError('Unexpected ConvGRU input channels.')
        if state is None:
            state = torch.zeros_like(inputs)
        if state.shape != inputs.shape:
            raise ValueError('ConvGRU state shape does not match inputs.')
        reset, update = torch.sigmoid(
            self.gates(torch.cat((inputs, state), dim=1))
        ).chunk(2, dim=1)
        candidate = torch.tanh(
            self.candidate(torch.cat((inputs, reset * state), dim=1))
        )
        return (1.0 - update) * state + update * candidate


class BidirectionalTemporalMemoryNet(nn.Module):
    """P23 U-Net with a zero-initialized bidirectional temporal residual.

    Every temporal step first receives the original P23 local context stack.
    A pair of ConvGRU cells then propagates low-resolution evidence forward
    and backward through a sequence.  The residual projection is initialized
    to zero, allowing a P23 checkpoint to be loaded without changing its
    initial predictions before memory training begins.
    """

    def __init__(
        self,
        input_channels,
        width=16,
        normalization_max_groups=8,
        temporal_attention_enabled=False,
        temporal_attention_num_heads=4,
        temporal_attention_output_init_std=0.0,
        temporal_attention_relative_bias_enabled=False,
        temporal_attention_relative_bias_max_distance=8,
        advection_alignment_enabled=False,
        advection_max_flow=0.0,
        fine_temporal_memory_enabled=False,
        fine_advection_max_flow=0.0,
        local_temporal_context_enabled=False,
        local_temporal_context_kernel_size=11,
        density_calibration_enabled=False,
        temporal_diff_enabled=False,
        ssa_enabled=False,
        confidence_head_enabled=False,
        target_center_enabled=False,
        target_level_enabled=False,
        target_level_downsample=4,
        objectness_gate_enabled=False,
        objectness_gate_strength=0.5,
        objectness_gate_downsample=4,
        center_memory_enabled=False,
        center_memory_channels=4,
        center_memory_downsample=4,
    ):
        super().__init__()
        if bool(target_center_enabled) and bool(confidence_head_enabled):
            raise ValueError(
                'Target-centre memory and confidence calibration cannot be '
                'enabled together.'
            )
        if bool(objectness_gate_enabled) and (
            bool(confidence_head_enabled)
            or bool(target_center_enabled)
            or bool(target_level_enabled)
        ):
            raise ValueError(
                'Objectness gating is mutually exclusive with the existing '
                'auxiliary output branches.'
            )
        if bool(center_memory_enabled) and not bool(target_center_enabled):
            raise ValueError('Center memory requires the target-centre head.')
        normalization_max_groups = int(normalization_max_groups)
        temporal_attention_num_heads = int(temporal_attention_num_heads)
        if normalization_max_groups <= 0:
            raise ValueError('normalization_max_groups must be positive.')
        if temporal_attention_num_heads <= 0:
            raise ValueError('temporal_attention_num_heads must be positive.')
        bottleneck_channels = int(width) * 6
        if bottleneck_channels % temporal_attention_num_heads != 0:
            raise ValueError(
                'Temporal attention channels must divide evenly into its heads.'
            )
        self.normalization_max_groups = normalization_max_groups
        self.temporal_attention_num_heads = temporal_attention_num_heads
        self.base = TemporalFrameNet(
            input_channels=int(input_channels),
            width=int(width),
            local_temporal_context_channels=(
                1 if bool(local_temporal_context_enabled) else 0
            ),
            density_calibration_enabled=bool(density_calibration_enabled),
            temporal_diff_enabled=bool(temporal_diff_enabled),
            ssa_enabled=bool(ssa_enabled),
            confidence_head_enabled=bool(confidence_head_enabled),
            target_center_enabled=bool(target_center_enabled),
            normalization_max_groups=normalization_max_groups,
        )
        self.local_temporal_context_enabled = bool(local_temporal_context_enabled)
        self.temporal_diff_enabled = bool(temporal_diff_enabled)
        self.ssa_enabled = bool(ssa_enabled)
        self.local_temporal_context_kernel_size = int(
            local_temporal_context_kernel_size
        )
        if (
            self.local_temporal_context_kernel_size <= 0
            or self.local_temporal_context_kernel_size % 2 == 0
        ):
            raise ValueError(
                'local_temporal_context_kernel_size must be a positive odd integer.'
            )
        self.confidence_head_enabled = bool(confidence_head_enabled)
        self.objectness_gate_enabled = bool(objectness_gate_enabled)
        self.objectness_gate_strength = float(objectness_gate_strength)
        self.objectness_gate_downsample = int(objectness_gate_downsample)
        if self.objectness_gate_enabled:
            if self.objectness_gate_strength <= 0.0:
                raise ValueError('Objectness gate strength must be positive.')
            if self.objectness_gate_downsample <= 0:
                raise ValueError('Objectness gate downsample must be positive.')
        self.objectness_center_head = None
        self.objectness_presence_head = None
        self.objectness_velocity_head = None
        self.objectness_event_gate = None
        if self.objectness_gate_enabled:
            hidden_width = max(4, int(width) // 2)
            self.objectness_center_head = nn.Sequential(
                nn.Conv2d(int(width), int(width), kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(int(width), 1, kernel_size=1),
            )
            self.objectness_presence_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(int(width), hidden_width),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_width, 1),
            )
            self.objectness_velocity_head = nn.Sequential(
                nn.Conv2d(int(width), int(width), kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(int(width), 2, kernel_size=1),
            )
            self.objectness_event_gate = nn.Sequential(
                nn.Conv2d(int(width) + 1, int(width), kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(int(width), 1, kernel_size=1),
            )
            # Keep the released M26 logits unchanged until this branch learns
            # a residual, while its supervised objectness heads remain active.
            nn.init.zeros_(self.objectness_event_gate[-1].weight)
            nn.init.zeros_(self.objectness_event_gate[-1].bias)
        # M91 is an auxiliary target-level task.  It never feeds the event
        # head, so attaching it to an M26 checkpoint preserves the released
        # event predictions before training while still shaping shared
        # decoder features through multi-task supervision.
        self.target_level_enabled = bool(target_level_enabled)
        self.target_level_downsample = int(target_level_downsample)
        if self.target_level_downsample <= 0:
            raise ValueError('target_level_downsample must be positive.')
        self.target_level_center_head = None
        self.target_level_presence_head = None
        self.target_level_velocity_head = None
        if self.target_level_enabled:
            hidden_width = max(4, int(width) // 2)
            self.target_level_center_head = nn.Sequential(
                nn.Conv2d(int(width), int(width), kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(int(width), 1, kernel_size=1),
            )
            self.target_level_presence_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(int(width), hidden_width),
                nn.ReLU(inplace=False),
                nn.Linear(hidden_width, 1),
            )
            self.target_level_velocity_head = nn.Sequential(
                nn.Conv2d(int(width), int(width), kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(int(width), 2, kernel_size=1),
            )
        self.forward_memory = ConvGRUCell(bottleneck_channels)
        self.backward_memory = ConvGRUCell(bottleneck_channels)
        self.memory_projection = nn.Conv2d(
            bottleneck_channels * 2,
            bottleneck_channels,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.memory_projection.weight)
        nn.init.zeros_(self.memory_projection.bias)

        self.temporal_attention_enabled = bool(temporal_attention_enabled)
        self.temporal_attn = None
        if self.temporal_attention_enabled:
            self.temporal_attn = TemporalSelfAttentionMemory(
                channels=bottleneck_channels,
                num_heads=temporal_attention_num_heads,
                pool_size=16,
                output_init_std=temporal_attention_output_init_std,
                relative_bias_enabled=temporal_attention_relative_bias_enabled,
                relative_bias_max_distance=temporal_attention_relative_bias_max_distance,
            )

        self.advection_alignment_enabled = bool(advection_alignment_enabled)
        self.advection_max_flow = float(advection_max_flow)
        self.flow_head = None
        if self.advection_alignment_enabled:
            self.flow_head = FlowEstimationHead(
                bottleneck_channels,
                max_displacement=self.advection_max_flow,
                normalization_max_groups=normalization_max_groups,
            )
        self._last_advection_consistency_loss = None
        self._last_advection_forward_flows = None

        # M35: a finer H/4 skip-memory path. At H/8, the typical 4 px target
        # motion is sub-cell; this branch keeps a one-cell motion observable
        # while its zero projection preserves an attached M26 checkpoint.
        self.fine_temporal_memory_enabled = bool(fine_temporal_memory_enabled)
        self.fine_advection_max_flow = float(fine_advection_max_flow)
        if self.fine_advection_max_flow < 0.0:
            raise ValueError('fine_advection_max_flow must be non-negative.')
        self.fine_forward_memory = None
        self.fine_backward_memory = None
        self.fine_memory_projection = None
        self.fine_flow_head = None
        if self.fine_temporal_memory_enabled:
            fine_channels = int(width) * 4
            self.fine_forward_memory = ConvGRUCell(fine_channels)
            self.fine_backward_memory = ConvGRUCell(fine_channels)
            self.fine_memory_projection = nn.Conv2d(
                fine_channels * 2,
                fine_channels,
                kernel_size=1,
                bias=True,
            )
            nn.init.zeros_(self.fine_memory_projection.weight)
            nn.init.zeros_(self.fine_memory_projection.bias)
            self.fine_flow_head = FlowEstimationHead(
                fine_channels,
                max_displacement=self.fine_advection_max_flow,
                normalization_max_groups=normalization_max_groups,
            )
        self._last_fine_advection_forward_flows = None

        # M48: an intentionally narrow temporal branch over target-centre
        # evidence, rather than generic H/4 features. Pooling makes the extra
        # ConvGRUs inexpensive while the zero output layers preserve M26
        # predictions exactly when this branch is first attached.
        self.center_memory_enabled = bool(center_memory_enabled)
        self.center_memory_channels = int(center_memory_channels)
        self.center_memory_downsample = int(center_memory_downsample)
        self.center_memory_projection_in = None
        self.center_memory_forward = None
        self.center_memory_backward = None
        self.center_memory_projection_out = None
        self.center_event_projection = None
        if self.center_memory_enabled:
            if self.center_memory_channels <= 0:
                raise ValueError('center_memory_channels must be positive.')
            if self.center_memory_downsample < 2:
                raise ValueError('center_memory_downsample must be at least two.')
            self.center_memory_projection_in = nn.Conv2d(
                1,
                self.center_memory_channels,
                kernel_size=1,
                bias=True,
            )
            self.center_memory_forward = ConvGRUCell(self.center_memory_channels)
            self.center_memory_backward = ConvGRUCell(self.center_memory_channels)
            self.center_memory_projection_out = nn.Conv2d(
                self.center_memory_channels * 2,
                1,
                kernel_size=1,
                bias=True,
            )
            self.center_event_projection = nn.Conv2d(
                1,
                1,
                kernel_size=1,
                bias=True,
            )
            nn.init.zeros_(self.center_memory_projection_out.weight)
            nn.init.zeros_(self.center_memory_projection_out.bias)
            nn.init.zeros_(self.center_event_projection.weight)
            nn.init.zeros_(self.center_event_projection.bias)

    @property
    def input_channels(self):
        return self.base.input_channels

    @property
    def total_input_channels(self):
        return self.base.total_input_channels

    def _encode(self, frames):
        if frames.ndim != 4:
            raise ValueError('frames must have shape [B, C, H, W].')
        if frames.shape[1] != self.total_input_channels:
            raise ValueError(
                'frames have {} channels, expected {}.'.format(
                    frames.shape[1], self.total_input_channels
                )
            )
        level0, level1, level2, bottleneck = self.base.encode_features(frames)
        return level0, level1, level2, bottleneck

    def encode_bottleneck(self, frames):
        """Encode a frame batch for full-stream inference memory passes."""
        return self._encode(frames)[-1]

    def encode_memory_features(self, frames):
        """Return H/4 and H/8 features for optional full-stream memory paths."""
        _, _, level2, bottleneck = self._encode(frames)
        return level2, bottleneck

    def _memory_residual(self, bottlenecks):
        if bottlenecks.ndim != 5:
            raise ValueError('bottlenecks must have shape [B, T, C, H, W].')
        batch_size, sequence_length = bottlenecks.shape[:2]
        if sequence_length <= 0:
            raise ValueError('Temporal sequence must not be empty.')

        self._last_advection_consistency_loss = bottlenecks.new_zeros(())
        consistency_terms = []
        # M52 preserves the released M26 representation in eval mode and
        # optimizes only the existing flow estimator. Its module mode decides
        # whether per-bin flow tensors need to be retained for supervision.
        collect_forward_flows = self.advection_alignment_enabled and (
            self.training or self.flow_head.training
        )
        self._last_advection_forward_flows = (
            [None] * sequence_length if collect_forward_flows else None
        )
        exact_advection_identity = (
            self.advection_alignment_enabled and self.flow_head.has_zero_output()
        )
        forward_states = []
        state = None
        previous = None
        for time_index in range(sequence_length):
            current = bottlenecks[:, time_index]
            if self.advection_alignment_enabled and state is not None:
                flow = self.flow_head(previous, current)
                if collect_forward_flows:
                    self._last_advection_forward_flows[time_index] = flow
                state = advect_warp(
                    state, flow, exact_identity=exact_advection_identity
                )
                consistency_terms.append(
                    advection_consistency_loss(
                        previous,
                        current,
                        flow,
                        exact_identity=exact_advection_identity,
                    )
                )
            state = self.forward_memory(current, state)
            forward_states.append(state)
            previous = current

        backward_states = [None] * sequence_length
        state = None
        previous = None
        for time_index in range(sequence_length - 1, -1, -1):
            current = bottlenecks[:, time_index]
            if self.advection_alignment_enabled and state is not None:
                flow = self.flow_head(previous, current)
                state = advect_warp(
                    state, flow, exact_identity=exact_advection_identity
                )
                consistency_terms.append(
                    advection_consistency_loss(
                        previous,
                        current,
                        flow,
                        exact_identity=exact_advection_identity,
                    )
                )
            state = self.backward_memory(current, state)
            backward_states[time_index] = state
            previous = current

        if consistency_terms:
            self._last_advection_consistency_loss = torch.stack(
                consistency_terms
            ).mean()

        memory_features = torch.cat(
            (
                torch.stack(forward_states, dim=1),
                torch.stack(backward_states, dim=1),
            ),
            dim=2,
        )
        flat_features = memory_features.reshape(
            batch_size * sequence_length,
            memory_features.shape[2],
            memory_features.shape[3],
            memory_features.shape[4],
        )
        projected = self.memory_projection(flat_features)
        residual = projected.reshape(
            batch_size,
            sequence_length,
            projected.shape[1],
            projected.shape[2],
            projected.shape[3],
        )

        if self.temporal_attention_enabled:
            attn_residual = self.temporal_attn(bottlenecks)
            residual = residual + attn_residual

        return residual

    def temporal_residual(self, bottlenecks):
        """Return one zero-initialized temporal residual per bottleneck map."""
        if bottlenecks.ndim == 4:
            return self._memory_residual(bottlenecks.unsqueeze(0)).squeeze(0)
        return self._memory_residual(bottlenecks)

    def _fine_memory_residual(self, level2_features):
        """Return motion-aligned H/4 skip residuals for M35."""
        if not self.fine_temporal_memory_enabled:
            raise RuntimeError('Fine temporal memory is disabled.')
        if level2_features.ndim != 5:
            raise ValueError('level2_features must have shape [B, T, C, H, W].')
        batch_size, sequence_length = level2_features.shape[:2]
        if sequence_length <= 0:
            raise ValueError('Temporal sequence must not be empty.')

        # M40 keeps the released M26 path in eval mode but trains this newly
        # attached branch. The fine module state, rather than the top-level
        # model state, therefore decides whether target-flow supervision needs
        # the per-bin flow tensors.
        collect_forward_flows = self.training or self.fine_flow_head.training
        self._last_fine_advection_forward_flows = (
            [None] * sequence_length if collect_forward_flows else None
        )
        exact_identity = self.fine_flow_head.has_zero_output()
        forward_states = []
        state = None
        previous = None
        for time_index in range(sequence_length):
            current = level2_features[:, time_index]
            if state is not None:
                flow = self.fine_flow_head(previous, current)
                if collect_forward_flows:
                    self._last_fine_advection_forward_flows[time_index] = flow
                state = advect_warp(state, flow, exact_identity=exact_identity)
            state = self.fine_forward_memory(current, state)
            forward_states.append(state)
            previous = current

        backward_states = [None] * sequence_length
        state = None
        previous = None
        for time_index in range(sequence_length - 1, -1, -1):
            current = level2_features[:, time_index]
            if state is not None:
                flow = self.fine_flow_head(previous, current)
                state = advect_warp(state, flow, exact_identity=exact_identity)
            state = self.fine_backward_memory(current, state)
            backward_states[time_index] = state
            previous = current

        memory_features = torch.cat(
            (
                torch.stack(forward_states, dim=1),
                torch.stack(backward_states, dim=1),
            ),
            dim=2,
        )
        flat_features = memory_features.reshape(
            batch_size * sequence_length,
            memory_features.shape[2],
            memory_features.shape[3],
            memory_features.shape[4],
        )
        projected = self.fine_memory_projection(flat_features)
        return projected.reshape(
            batch_size,
            sequence_length,
            projected.shape[1],
            projected.shape[2],
            projected.shape[3],
        )

    def fine_temporal_residual(self, level2_features):
        """Return one zero-initialized M35 residual per H/4 skip feature."""
        if level2_features.ndim == 4:
            return self._fine_memory_residual(
                level2_features.unsqueeze(0)
            ).squeeze(0)
        return self._fine_memory_residual(level2_features)

    def _center_memory_residual(self, center_logits):
        """Aggregate full-stream centre evidence at a compact spatial scale."""
        if not self.center_memory_enabled:
            raise RuntimeError('Center memory is disabled.')
        if center_logits.ndim != 5 or center_logits.shape[2] != 1:
            raise ValueError(
                'center_logits must have shape [B, T, 1, H, W].'
            )
        batch_size, sequence_length, _, height, width = center_logits.shape
        if sequence_length <= 0:
            raise ValueError('Temporal sequence must not be empty.')
        flat_logits = center_logits.reshape(
            batch_size * sequence_length,
            1,
            height,
            width,
        )
        pooled_logits = F.avg_pool2d(
            flat_logits,
            kernel_size=self.center_memory_downsample,
            stride=self.center_memory_downsample,
            ceil_mode=True,
        )
        projected = self.center_memory_projection_in(pooled_logits).reshape(
            batch_size,
            sequence_length,
            self.center_memory_channels,
            pooled_logits.shape[2],
            pooled_logits.shape[3],
        )

        forward_states = []
        state = None
        for time_index in range(sequence_length):
            state = self.center_memory_forward(
                projected[:, time_index],
                state,
            )
            forward_states.append(state)

        backward_states = [None] * sequence_length
        state = None
        for time_index in range(sequence_length - 1, -1, -1):
            state = self.center_memory_backward(
                projected[:, time_index],
                state,
            )
            backward_states[time_index] = state

        memory_features = torch.cat(
            (
                torch.stack(forward_states, dim=1),
                torch.stack(backward_states, dim=1),
            ),
            dim=2,
        )
        flat_residual = self.center_memory_projection_out(
            memory_features.reshape(
                batch_size * sequence_length,
                memory_features.shape[2],
                memory_features.shape[3],
                memory_features.shape[4],
            )
        )
        full_residual = _interpolate(
            flat_residual,
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        )
        return full_residual.reshape(
            batch_size,
            sequence_length,
            1,
            height,
            width,
        )

    def center_temporal_residual(self, center_logits):
        """Return a centre-logit residual for [T,...] or [B,T,...] inputs."""
        if center_logits.ndim == 4:
            return self._center_memory_residual(center_logits.unsqueeze(0)).squeeze(0)
        return self._center_memory_residual(center_logits)

    def _objectness_outputs(self, decoded0):
        if not self.objectness_gate_enabled:
            raise ValueError('Objectness gating is disabled.')
        objectness_features = decoded0
        if self.objectness_gate_downsample > 1:
            objectness_features = F.avg_pool2d(
                objectness_features,
                kernel_size=self.objectness_gate_downsample,
                stride=self.objectness_gate_downsample,
                ceil_mode=False,
            )
        center_logits = self.objectness_center_head(objectness_features)
        presence_logits = self.objectness_presence_head(objectness_features).squeeze(-1)
        velocity = self.objectness_velocity_head(objectness_features)
        center_probability = _interpolate(
            torch.sigmoid(center_logits),
            size=decoded0.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        gated_features = torch.cat((decoded0, center_probability), dim=1)
        event_delta = self.objectness_gate_strength * torch.tanh(
            self.objectness_event_gate(gated_features)
        )
        return event_delta, center_logits, presence_logits, velocity

    def _decode(
        self,
        level0,
        level1,
        level2,
        bottleneck,
        base_input=None,
        return_confidence_logits=False,
        return_target_center_logits=False,
        return_target_level_outputs=False,
        return_objectness_outputs=False,
        center_memory_residual=None,
    ):
        decoded2 = self.base.decoder2(bottleneck, level2)
        decoded1 = self.base.decoder1(decoded2, level1)
        decoded0 = self.base.decoder0(decoded1, level0)
        if self.base.density_calibration_enabled:
            if base_input is None:
                raise ValueError(
                    'density calibration requires base_input in _decode.'
                )
            decoded0 = self.base.density_calibrator(decoded0, base_input)
        if self.base.ssa_enabled:
            if base_input is None:
                raise ValueError('SSA requires base_input in _decode.')
            decoded0 = self.base.apply_ssa(decoded0, base_input)
        logits = self.base.head(decoded0)
        objectness_outputs = None
        if self.objectness_gate_enabled:
            objectness_outputs = self._objectness_outputs(decoded0)
            logits = logits + objectness_outputs[0]
        target_level_outputs = None
        if self.target_level_enabled and return_target_level_outputs:
            target_level_features = decoded0
            if self.target_level_downsample > 1:
                target_level_features = F.avg_pool2d(
                    decoded0,
                    kernel_size=self.target_level_downsample,
                    stride=self.target_level_downsample,
                    ceil_mode=False,
                )
            target_level_outputs = (
                self.target_level_center_head(target_level_features),
                self.target_level_presence_head(target_level_features).squeeze(-1),
                self.target_level_velocity_head(target_level_features),
            )
        target_center_logits = None
        if self.base.target_center_enabled:
            target_center_logits = self.base.target_center_head(decoded0)
            if center_memory_residual is not None:
                if center_memory_residual.shape != target_center_logits.shape:
                    raise ValueError(
                        'Center-memory residual does not match target-centre logits.'
                    )
                if not self.center_memory_enabled:
                    raise ValueError('Center memory is disabled.')
                target_center_logits = target_center_logits + center_memory_residual
            logits = logits + self.base.target_center_residual(target_center_logits)
            if center_memory_residual is not None:
                logits = logits + self.center_event_projection(center_memory_residual)

        confidence_logits = None
        if self.confidence_head_enabled and return_confidence_logits:
            confidence_logits = self.base.confidence_head(decoded0)
        if return_target_center_logits and return_confidence_logits:
            return logits, target_center_logits, confidence_logits
        if return_target_level_outputs:
            if not self.target_level_enabled:
                raise ValueError(
                    'target-level outputs were requested from a model without '
                    'the target-level branch.'
                )
            return (logits,) + target_level_outputs
        if return_objectness_outputs:
            if not self.objectness_gate_enabled:
                raise ValueError(
                    'Objectness outputs were requested from a model without '
                    'the objectness gate.'
                )
            return (logits,) + objectness_outputs[1:]
        if return_target_center_logits:
            return logits, target_center_logits
        if return_confidence_logits:
            return logits, confidence_logits
        return logits

    def decode_with_residual(
        self,
        frames,
        residual,
        fine_residual=None,
        return_confidence_logits=False,
        return_target_center_logits=False,
        return_target_level_outputs=False,
        return_objectness_outputs=False,
        center_memory_residual=None,
    ):
        """Decode a frame batch after a full-stream memory pass."""
        level0, level1, level2, bottleneck = self._encode(frames)
        if residual.shape != bottleneck.shape:
            raise ValueError('Temporal residual does not match bottleneck shape.')
        if fine_residual is not None and fine_residual.shape != level2.shape:
            raise ValueError('Fine temporal residual does not match level2 shape.')
        if fine_residual is not None:
            level2 = level2 + fine_residual
        return self._decode(
            level0,
            level1,
            level2,
            bottleneck + residual,
            base_input=frames[:, :self.input_channels],
            return_confidence_logits=return_confidence_logits,
            return_target_center_logits=return_target_center_logits,
            return_target_level_outputs=return_target_level_outputs,
            return_objectness_outputs=return_objectness_outputs,
            center_memory_residual=center_memory_residual,
        )

    def decode_features_with_residual(
        self,
        frames,
        residual,
        fine_residual=None,
    ):
        """Return the read-only decoder feature used by an auxiliary head.

        This deliberately stops immediately before ``base.head`` and all
        event-logit residual branches.  It is used by M95's independently
        trained frozen centre proposer; the released event prediction path
        remains exactly the one implemented by :meth:`decode_with_residual`.
        """
        level0, level1, level2, bottleneck = self._encode(frames)
        if residual.shape != bottleneck.shape:
            raise ValueError('Temporal residual does not match bottleneck shape.')
        if fine_residual is not None and fine_residual.shape != level2.shape:
            raise ValueError('Fine temporal residual does not match level2 shape.')
        if fine_residual is not None:
            level2 = level2 + fine_residual
        decoded2 = self.base.decoder2(bottleneck + residual, level2)
        decoded1 = self.base.decoder1(decoded2, level1)
        decoded0 = self.base.decoder0(decoded1, level0)
        if self.base.density_calibration_enabled:
            decoded0 = self.base.density_calibrator(
                decoded0,
                frames[:, :self.input_channels],
            )
        return decoded0

    def forward(
        self,
        frames,
        return_target_center_logits=False,
        return_target_level_outputs=False,
        return_objectness_outputs=False,
    ):
        """Predict logit maps for ``[B, T, C, H, W]`` temporal sequences."""
        if frames.ndim != 5:
            raise ValueError('frames must have shape [B, T, C, H, W].')
        batch_size, sequence_length = frames.shape[:2]
        if sequence_length <= 0:
            raise ValueError('Temporal sequence must not be empty.')
        flat_frames = frames.reshape(
            batch_size * sequence_length,
            frames.shape[2],
            frames.shape[3],
            frames.shape[4],
        )
        level0, level1, level2, bottleneck = self._encode(flat_frames)
        bottleneck = bottleneck.reshape(
            batch_size,
            sequence_length,
            bottleneck.shape[1],
            bottleneck.shape[2],
            bottleneck.shape[3],
        )
        residual = self._memory_residual(bottleneck).reshape_as(
            bottleneck.reshape(
                batch_size * sequence_length,
                bottleneck.shape[2],
                bottleneck.shape[3],
                bottleneck.shape[4],
            )
        )
        fine_residual = None
        if self.fine_temporal_memory_enabled:
            level2_sequence = level2.reshape(
                batch_size,
                sequence_length,
                level2.shape[1],
                level2.shape[2],
                level2.shape[3],
            )
            fine_residual = self._fine_memory_residual(level2_sequence).reshape_as(
                level2
            )
        decode_output = self._decode(
            level0,
            level1,
            level2 if fine_residual is None else level2 + fine_residual,
            bottleneck.reshape(
                batch_size * sequence_length,
                bottleneck.shape[2],
                bottleneck.shape[3],
                bottleneck.shape[4],
            ) + residual,
            base_input=flat_frames[:, :self.input_channels],
            return_confidence_logits=self.confidence_head_enabled,
            return_target_center_logits=self.base.target_center_enabled,
            return_target_level_outputs=return_target_level_outputs,
            return_objectness_outputs=return_objectness_outputs,
        )
        if self.objectness_gate_enabled and return_objectness_outputs:
            (
                logits,
                objectness_center_logits,
                objectness_presence_logits,
                objectness_velocity,
            ) = decode_output
            return (
                logits.reshape(batch_size, sequence_length, *logits.shape[1:]),
                objectness_center_logits.reshape(
                    batch_size,
                    sequence_length,
                    *objectness_center_logits.shape[1:],
                ),
                objectness_presence_logits.reshape(batch_size, sequence_length),
                objectness_velocity.reshape(
                    batch_size,
                    sequence_length,
                    *objectness_velocity.shape[1:],
                ),
            )
        if self.target_level_enabled and return_target_level_outputs:
            (
                logits,
                target_level_center_logits,
                target_level_presence_logits,
                target_level_velocity,
            ) = decode_output
            logits = logits.reshape(
                batch_size,
                sequence_length,
                logits.shape[1],
                logits.shape[2],
                logits.shape[3],
            )
            target_level_center_logits = target_level_center_logits.reshape(
                batch_size,
                sequence_length,
                target_level_center_logits.shape[1],
                target_level_center_logits.shape[2],
                target_level_center_logits.shape[3],
            )
            target_level_presence_logits = target_level_presence_logits.reshape(
                batch_size,
                sequence_length,
            )
            target_level_velocity = target_level_velocity.reshape(
                batch_size,
                sequence_length,
                target_level_velocity.shape[1],
                target_level_velocity.shape[2],
                target_level_velocity.shape[3],
            )
            return (
                logits,
                target_level_center_logits,
                target_level_presence_logits,
                target_level_velocity,
            )
        if self.base.target_center_enabled:
            logits, target_center_logits = decode_output
            target_center_logits = target_center_logits.reshape(
                batch_size,
                sequence_length,
                target_center_logits.shape[1],
                target_center_logits.shape[2],
                target_center_logits.shape[3],
            )
            if self.center_memory_enabled:
                center_residual = self._center_memory_residual(target_center_logits)
                logits = logits + self.base.target_center_residual(
                    center_residual.reshape_as(logits)
                ) + self.center_event_projection(
                    center_residual.reshape_as(logits)
                )
                target_center_logits = target_center_logits + center_residual
            logits = logits.reshape(
                batch_size,
                sequence_length,
                logits.shape[1],
                logits.shape[2],
                logits.shape[3],
            )
            if return_target_center_logits:
                return logits, target_center_logits
            return logits
        if self.confidence_head_enabled:
            logits, confidence_logits = decode_output
            logits = logits.reshape(
                batch_size,
                sequence_length,
                logits.shape[1],
                logits.shape[2],
                logits.shape[3],
            )
            confidence_logits = confidence_logits.reshape(
                batch_size,
                sequence_length,
                confidence_logits.shape[1],
                confidence_logits.shape[2],
                confidence_logits.shape[3],
            )
            return logits, confidence_logits
        logits = decode_output
        return logits.reshape(
            batch_size,
            sequence_length,
            logits.shape[1],
            logits.shape[2],
            logits.shape[3],
        )
