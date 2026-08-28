"""A compact full-frame temporal event segmentation network."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as functional

from model.modules.confidence_head import ConfidenceHead
from utils.multiscale_motion import (
    build_multiscale_motion_persistence_channels,
    multiscale_motion_channel_count,
)


def _group_count(channels, maximum_groups=8):
    channels = int(channels)
    for group_count in range(min(int(maximum_groups), channels), 0, -1):
        if channels % group_count == 0:
            return group_count
    return 1


def _interpolate(input_tensor, **kwargs):
    """Keep pair-audit interpolation off CUDA's nondeterministic backward."""
    if (
        input_tensor.is_cuda
        and os.environ.get('EVSOD_DETERMINISTIC_WARP_CPU', '').strip() == '1'
    ):
        return functional.interpolate(input_tensor.cpu(), **kwargs).to(
            input_tensor.device
        )
    return functional.interpolate(input_tensor, **kwargs)


def append_local_contrast_channels(inputs, kernel_size=9):
    """Append edge-padded local contrast channels on the active device."""
    if inputs.ndim != 4:
        raise ValueError('inputs must have shape [B, C, H, W].')
    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError('kernel_size must be a positive odd integer.')
    radius = kernel_size // 2
    local_mean = functional.avg_pool2d(
        functional.pad(
            inputs,
            (radius, radius, radius, radius),
            mode='replicate',
        ),
        kernel_size=kernel_size,
        stride=1,
    )
    return torch.cat((inputs, inputs - local_mean), dim=1)


def build_motion_persistence_channels(
    inputs,
    context_bins,
    spatial_radius_per_bin=4,
):
    """Measure centre-bin activity supported by shifted neighbouring bins.

    The source stack is arranged as negative/positive polarity pairs for each
    temporal bin. For every non-centre bin, a local maximum allows a moving
    event pattern to shift by ``spatial_radius_per_bin`` pixels per bin before
    it is intersected with the centre-bin activity. The output therefore has
    one non-negative channel for each neighbouring temporal bin.
    """
    if inputs.ndim != 4:
        raise ValueError('inputs must have shape [B, C, H, W].')
    context_bins = int(context_bins)
    spatial_radius_per_bin = int(spatial_radius_per_bin)
    if context_bins < 1 or context_bins % 2 == 0:
        raise ValueError('context_bins must be a positive odd integer.')
    if inputs.shape[1] != context_bins * 2:
        raise ValueError(
            'inputs have {} channels, expected {} raw temporal channels.'.format(
                inputs.shape[1],
                context_bins * 2,
            )
        )
    if spatial_radius_per_bin < 0:
        raise ValueError('spatial_radius_per_bin must be non-negative.')

    centre_bin = context_bins // 2
    centre_start = centre_bin * 2
    centre_activity = inputs[:, centre_start:centre_start + 2].sum(
        dim=1,
        keepdim=True,
    )
    features = []
    for temporal_bin in range(context_bins):
        if temporal_bin == centre_bin:
            continue
        neighbour_start = temporal_bin * 2
        neighbour_activity = inputs[:, neighbour_start:neighbour_start + 2].sum(
            dim=1,
            keepdim=True,
        )
        radius = spatial_radius_per_bin * abs(temporal_bin - centre_bin)
        if radius:
            neighbour_activity = functional.max_pool2d(
                neighbour_activity,
                kernel_size=radius * 2 + 1,
                stride=1,
                padding=radius,
            )
        features.append(torch.minimum(centre_activity, neighbour_activity))
    if not features:
        return inputs[:, :0]
    return torch.cat(features, dim=1)


def build_polarity_temporal_diff_channels(inputs):
    """Build label-free polarity-aware temporal difference channels.

    The raw context is stored as negative/positive pairs for each temporal
    bin, with the centre bin in the middle.  The returned eight channels
    expose short past/future changes, a longer past trend, polarity imbalance,
    and total activity change.  They are derived only from the input frame so
    the branch remains valid for the unlabeled public test videos.
    """
    if inputs.ndim != 4:
        raise ValueError('inputs must have shape [B, C, H, W].')
    channel_count = int(inputs.shape[1])
    if channel_count < 10 or channel_count % 2 != 0:
        raise ValueError('inputs must have an even channel count of at least 10.')
    context_bins = channel_count // 2
    centre_bin = context_bins // 2
    centre_start = centre_bin * 2
    centre_negative = inputs[:, centre_start]
    centre_positive = inputs[:, centre_start + 1]
    features = []
    for relative_bin in (-1, 1):
        neighbour_start = (centre_bin + relative_bin) * 2
        features.append(centre_negative - inputs[:, neighbour_start])
        features.append(centre_positive - inputs[:, neighbour_start + 1])
    long_past_start = (centre_bin - 2) * 2
    features.append(centre_negative - inputs[:, long_past_start])
    features.append(centre_positive - inputs[:, long_past_start + 1])
    features.append(centre_positive - centre_negative)
    features.append(
        (centre_positive + centre_negative)
        - (inputs[:, long_past_start] + inputs[:, long_past_start + 1])
    )
    return torch.stack(features, dim=1)


class ConvNormAct(nn.Module):
    """Three-by-three convolution with batch-size-independent normalization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        dilation=1,
        normalization_max_groups=8,
    ):
        super().__init__()
        dilation = int(dilation)
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(
                    out_channels,
                    maximum_groups=normalization_max_groups,
                ),
                out_channels,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class ResidualConvBlock(nn.Module):
    """A two-convolution residual block that preserves fine target detail."""

    def __init__(
        self,
        in_channels,
        out_channels,
        dilation=1,
        normalization_max_groups=8,
    ):
        super().__init__()
        self.first = ConvNormAct(
            in_channels,
            out_channels,
            dilation=dilation,
            normalization_max_groups=normalization_max_groups,
        )
        self.second = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=int(dilation),
                dilation=int(dilation),
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(
                    out_channels,
                    maximum_groups=normalization_max_groups,
                ),
                out_channels,
            ),
        )
        self.skip = (
            nn.Identity()
            if int(in_channels) == int(out_channels)
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.activation(self.second(self.first(inputs)) + self.skip(inputs))


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization_max_groups=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(
                    out_channels,
                    maximum_groups=normalization_max_groups,
                ),
                out_channels,
            ),
            nn.ReLU(inplace=True),
            ResidualConvBlock(
                out_channels,
                out_channels,
                normalization_max_groups=normalization_max_groups,
            ),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        normalization_max_groups=8,
    ):
        super().__init__()
        self.block = ResidualConvBlock(
            int(in_channels) + int(skip_channels),
            out_channels,
            normalization_max_groups=normalization_max_groups,
        )

    def forward(self, inputs, skip):
        inputs = _interpolate(
            inputs,
            size=skip.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        return self.block(torch.cat((inputs, skip), dim=1))


class DensityAdaptiveChannelCalibrator(nn.Module):
    """Density-adaptive channel calibrator.

    Design principles:
    1. Channel-only reweighting preserves spatial structure (protects IoU).
    2. Global density statistics drive channel importance (suppresses Fa).
    3. Global pooling + MLP similar to SE-Block, conditioned on density.
    4. Residual identity initialization (weights ≈ 1.0 at startup).
    """

    def __init__(self, feature_channels, reduction=16):
        super().__init__()
        self.feature_channels = feature_channels

        self.density_encoder = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.gating_network = nn.Sequential(
            nn.Linear(1, feature_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels // reduction, feature_channels, bias=False),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.zeros_(self.gating_network[0].weight)
        nn.init.zeros_(self.gating_network[2].weight)
        if self.gating_network[2].bias is not None:
            nn.init.zeros_(self.gating_network[0].bias)
            self.gating_network[2].bias.data.fill_(4.0)

    def forward(self, features, original_input):
        """Args:
            features: Decoder output [B, C, H, W]
            original_input: Raw input frames [B, T*2, H, W]

        Returns:
            torch.Tensor: Calibrated features [B, C, H, W]
        """
        B, C, H, W = features.shape

        density_map = original_input.abs().sum(dim=1, keepdim=True)
        global_density = self.density_encoder(density_map).view(B, -1)
        channel_weights = self.gating_network(global_density).view(B, C, 1, 1)

        return features * channel_weights


class SmearAwareAlignment(nn.Module):
    """Small output-side residual for directional event smear.

    The branch is deliberately placed immediately before the event head.  Its
    final projection is zero-initialized, so attaching it to a released
    checkpoint is an exact identity until the branch is trained.
    """

    def __init__(self, feature_channels, use_diff=False, hidden=16, strip_kernel=7):
        super().__init__()
        feature_channels = int(feature_channels)
        hidden = int(hidden)
        strip_kernel = int(strip_kernel)
        if feature_channels <= 0 or hidden <= 0:
            raise ValueError('feature_channels and hidden must be positive.')
        if strip_kernel <= 0 or strip_kernel % 2 == 0:
            raise ValueError('strip_kernel must be a positive odd integer.')
        self.use_diff = bool(use_diff)
        input_channels = feature_channels + (8 if self.use_diff else 0)
        radius = strip_kernel // 2
        self.horizontal = nn.Conv2d(
            input_channels,
            hidden,
            kernel_size=(1, strip_kernel),
            padding=(0, radius),
            bias=False,
        )
        self.vertical = nn.Conv2d(
            input_channels,
            hidden,
            kernel_size=(strip_kernel, 1),
            padding=(radius, 0),
            bias=False,
        )
        self.square = nn.Conv2d(
            input_channels,
            hidden,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.merge = nn.Conv2d(hidden * 3, feature_channels, kernel_size=1, bias=True)
        self.residual = nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, decoded_features, diff_features=None):
        if self.use_diff and diff_features is None:
            raise ValueError('use_diff requires diff_features in forward.')
        if not self.use_diff and diff_features is not None:
            raise ValueError('diff_features supplied but use_diff is False.')
        inputs = decoded_features
        if self.use_diff:
            inputs = torch.cat((decoded_features, diff_features), dim=1)
        branches = torch.cat(
            (
                self.horizontal(inputs),
                self.vertical(inputs),
                self.square(inputs),
            ),
            dim=1,
        )
        merged = self.merge(functional.relu(branches, inplace=False))
        return self.residual(merged)


class TemporalFrameNet(nn.Module):
    """Predict a target logit for every pixel of a metric-time event frame."""

    def __init__(
        self,
        input_channels,
        width=16,
        local_contrast_channels=0,
        motion_persistence_channels=0,
        fine_detail_channels=0,
        local_temporal_context_channels=0,
        target_center_enabled=False,
        confidence_head_enabled=False,
        density_calibration_enabled=False,
        temporal_diff_enabled=False,
        ssa_enabled=False,
        normalization_max_groups=8,
    ):
        super().__init__()
        input_channels = int(input_channels)
        width = int(width)
        local_contrast_channels = int(local_contrast_channels)
        motion_persistence_channels = int(motion_persistence_channels)
        fine_detail_channels = int(fine_detail_channels)
        local_temporal_context_channels = int(local_temporal_context_channels)
        normalization_max_groups = int(normalization_max_groups)
        if input_channels <= 0 or width <= 0:
            raise ValueError('input_channels and width must be positive.')
        if local_contrast_channels < 0:
            raise ValueError('local_contrast_channels must not be negative.')
        if motion_persistence_channels < 0:
            raise ValueError('motion_persistence_channels must not be negative.')
        if fine_detail_channels < 0:
            raise ValueError('fine_detail_channels must not be negative.')
        if local_temporal_context_channels < 0:
            raise ValueError(
                'local_temporal_context_channels must not be negative.'
            )
        if normalization_max_groups <= 0:
            raise ValueError('normalization_max_groups must be positive.')

        self.input_channels = input_channels
        self.local_contrast_channels = local_contrast_channels
        self.motion_persistence_channels = motion_persistence_channels
        self.fine_detail_channels = fine_detail_channels
        self.local_temporal_context_channels = local_temporal_context_channels
        self.target_center_enabled = bool(target_center_enabled)
        self.confidence_head_enabled = bool(confidence_head_enabled)
        self.density_calibration_enabled = bool(density_calibration_enabled)
        self.temporal_diff_enabled = bool(temporal_diff_enabled)
        self.ssa_enabled = bool(ssa_enabled)
        self.normalization_max_groups = normalization_max_groups
        width2 = width * 2
        width4 = width * 4
        width6 = width * 6
        self.encoder0 = ResidualConvBlock(
            input_channels,
            width,
            normalization_max_groups=normalization_max_groups,
        )
        self.local_contrast_adapter = None
        if local_contrast_channels:
            self.local_contrast_adapter = nn.Conv2d(
                local_contrast_channels,
                width,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            nn.init.zeros_(self.local_contrast_adapter.weight)
            nn.init.zeros_(self.local_contrast_adapter.bias)
        self.motion_persistence_adapter = None
        if motion_persistence_channels:
            self.motion_persistence_adapter = nn.Conv2d(
                motion_persistence_channels,
                width,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            nn.init.zeros_(self.motion_persistence_adapter.weight)
            nn.init.zeros_(self.motion_persistence_adapter.bias)
        self.fine_detail_adapter = None
        if fine_detail_channels:
            self.fine_detail_adapter = nn.Conv2d(
                fine_detail_channels,
                width,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            nn.init.zeros_(self.fine_detail_adapter.weight)
            nn.init.zeros_(self.fine_detail_adapter.bias)
        self.local_temporal_context_adapter = None
        if local_temporal_context_channels:
            self.local_temporal_context_adapter = nn.Conv2d(
                local_temporal_context_channels,
                width,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            nn.init.zeros_(self.local_temporal_context_adapter.weight)
            nn.init.zeros_(self.local_temporal_context_adapter.bias)
        self.temporal_diff_adapter = None
        if self.temporal_diff_enabled:
            self.temporal_diff_adapter = nn.Conv2d(
                8,
                width,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            nn.init.zeros_(self.temporal_diff_adapter.weight)
            nn.init.zeros_(self.temporal_diff_adapter.bias)
        self.encoder1 = DownBlock(
            width,
            width2,
            normalization_max_groups=normalization_max_groups,
        )
        self.encoder2 = DownBlock(
            width2,
            width4,
            normalization_max_groups=normalization_max_groups,
        )
        self.encoder3 = DownBlock(
            width4,
            width6,
            normalization_max_groups=normalization_max_groups,
        )
        self.context = nn.Sequential(
            ResidualConvBlock(
                width6,
                width6,
                dilation=2,
                normalization_max_groups=normalization_max_groups,
            ),
            ResidualConvBlock(
                width6,
                width6,
                dilation=4,
                normalization_max_groups=normalization_max_groups,
            ),
        )
        self.decoder2 = UpBlock(
            width6,
            width4,
            width4,
            normalization_max_groups=normalization_max_groups,
        )
        self.decoder1 = UpBlock(
            width4,
            width2,
            width2,
            normalization_max_groups=normalization_max_groups,
        )
        self.decoder0 = UpBlock(
            width2,
            width,
            width,
            normalization_max_groups=normalization_max_groups,
        )
        self.head = nn.Conv2d(width, 1, kernel_size=1)

        # P32 keeps P23's event head intact at initialization. The centre head
        # receives dense train-only supervision, while this zero projection lets
        # event BCE decide how much centre evidence should affect each event.
        self.target_center_head = None
        self.target_center_residual = None
        if self.target_center_enabled:
            self.target_center_head = nn.Sequential(
                ConvNormAct(
                    width,
                    width,
                    normalization_max_groups=normalization_max_groups,
                ),
                nn.Conv2d(width, 1, kernel_size=1),
            )
            nn.init.zeros_(self.target_center_head[-1].weight)
            nn.init.zeros_(self.target_center_head[-1].bias)
            self.target_center_residual = nn.Conv2d(
                1,
                1,
                kernel_size=1,
                bias=False,
            )
            nn.init.zeros_(self.target_center_residual.weight)

        self.confidence_head = None
        if self.confidence_head_enabled:
            self.confidence_head = ConfidenceHead(
                width,
                normalization_max_groups=normalization_max_groups,
            )

        self.density_calibrator = None
        if self.density_calibration_enabled:
            self.density_calibrator = DensityAdaptiveChannelCalibrator(
                feature_channels=width,
                reduction=16,
            )
        self.ssa = None
        if self.ssa_enabled:
            self.ssa = SmearAwareAlignment(
                feature_channels=width,
                use_diff=self.temporal_diff_enabled,
            )

    @property
    def total_input_channels(self):
        """Number of raw and optional adapter input channels."""
        return (
            self.input_channels
            + self.local_contrast_channels
            + self.motion_persistence_channels
            + self.fine_detail_channels
            + self.local_temporal_context_channels
        )

    def encode_features(self, inputs):
        """Return encoder features shared by frame and memory inference.

        The local temporal-context adapter is zero-initialized.  Unlike older
        input adapters, it must not introduce a new activation at attachment
        time, otherwise a zero adapter would still perturb released logits.
        """
        if inputs.ndim != 4:
            raise ValueError('inputs must have shape [B, C, H, W].')
        if inputs.shape[1] != self.total_input_channels:
            raise ValueError(
                'inputs have {} channels, expected {}.'.format(
                    inputs.shape[1], self.total_input_channels
                )
            )
        level0 = self.encoder0(inputs[:, :self.input_channels])
        adapter_offset = self.input_channels
        legacy_adapter_enabled = False
        if self.local_contrast_adapter is not None:
            contrast_end = adapter_offset + self.local_contrast_channels
            level0 = level0 + self.local_contrast_adapter(
                inputs[:, adapter_offset:contrast_end]
            )
            adapter_offset = contrast_end
            legacy_adapter_enabled = True
        if self.motion_persistence_adapter is not None:
            motion_end = adapter_offset + self.motion_persistence_channels
            level0 = level0 + self.motion_persistence_adapter(
                inputs[:, adapter_offset:motion_end]
            )
            adapter_offset = motion_end
            legacy_adapter_enabled = True
        if self.fine_detail_adapter is not None:
            fine_detail_end = adapter_offset + self.fine_detail_channels
            level0 = level0 + self.fine_detail_adapter(
                inputs[:, adapter_offset:fine_detail_end]
            )
            adapter_offset = fine_detail_end
            legacy_adapter_enabled = True
        if self.local_temporal_context_adapter is not None:
            context_end = adapter_offset + self.local_temporal_context_channels
            level0 = level0 + self.local_temporal_context_adapter(
                inputs[:, adapter_offset:context_end]
            )
        if self.temporal_diff_adapter is not None:
            diff_inputs = build_polarity_temporal_diff_channels(
                inputs[:, :self.input_channels]
            )
            level0 = level0 + self.temporal_diff_adapter(diff_inputs)
            legacy_adapter_enabled = True
        if legacy_adapter_enabled:
            level0 = functional.relu(level0, inplace=False)
        level1 = self.encoder1(level0)
        level2 = self.encoder2(level1)
        level3 = self.context(self.encoder3(level2))
        return level0, level1, level2, level3

    def apply_ssa(self, decoded_features, base_input):
        """Apply the optional output-side residual before ``self.head``."""
        if self.ssa is None:
            return decoded_features
        diff_features = None
        if self.ssa.use_diff:
            diff_features = build_polarity_temporal_diff_channels(base_input)
        return decoded_features + self.ssa(decoded_features, diff_features)

    def forward(
        self,
        inputs,
        return_target_center_logits=False,
        return_confidence_logits=False,
    ):
        level0, level1, level2, level3 = self.encode_features(inputs)
        decoded2 = self.decoder2(level3, level2)
        decoded1 = self.decoder1(decoded2, level1)
        decoded0 = self.decoder0(decoded1, level0)
        if self.density_calibration_enabled:
            decoded0 = self.density_calibrator(
                decoded0,
                inputs[:, :self.input_channels],
            )
        decoded0 = self.apply_ssa(decoded0, inputs[:, :self.input_channels])
        logits = self.head(decoded0)

        confidence_logits = None
        if self.confidence_head_enabled:
            confidence_logits = self.confidence_head(decoded0)

        if self.target_center_enabled:
            target_center_logits = self.target_center_head(decoded0)
            logits = logits + self.target_center_residual(target_center_logits)
            if return_target_center_logits or return_confidence_logits:
                result = [logits]
                if return_target_center_logits:
                    result.append(target_center_logits)
                if return_confidence_logits:
                    result.append(confidence_logits)
                return tuple(result)
        elif return_target_center_logits or return_confidence_logits:
            if return_target_center_logits and not self.target_center_enabled:
                raise ValueError(
                    'target-center logits were requested from a model without '
                    'the target-center branch.'
                )
            result = [logits]
            if return_confidence_logits:
                result.append(confidence_logits)
            if return_target_center_logits:
                result.append(target_center_logits)
            return tuple(result)
        return logits


def gather_event_logits(logit_maps, event_batch_indices, event_y, event_x):
    """Extract one model logit for every original event coordinate."""
    if logit_maps.ndim != 4 or logit_maps.shape[1] != 1:
        raise ValueError('logit_maps must have shape [B, 1, H, W].')
    event_batch_indices = event_batch_indices.long()
    event_y = event_y.long()
    event_x = event_x.long()
    if not (
        event_batch_indices.shape
        == event_y.shape
        == event_x.shape
    ):
        raise ValueError('Event batch, y, and x tensors must have matching shapes.')
    if event_batch_indices.numel() == 0:
        return logit_maps.reshape(-1)[:0]
    if (
        event_batch_indices.min() < 0
        or event_batch_indices.max() >= logit_maps.shape[0]
        or event_y.min() < 0
        or event_y.max() >= logit_maps.shape[2]
        or event_x.min() < 0
        or event_x.max() >= logit_maps.shape[3]
    ):
        raise ValueError('An event coordinate is outside the logit-map bounds.')
    return logit_maps[
        event_batch_indices,
        0,
        event_y,
        event_x,
    ]
