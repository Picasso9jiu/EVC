"""Inference helpers for bidirectional full-stream temporal memory."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from dataset.temporal_frame import (
    build_temporal_context_frame,
    temporal_frame_video_from_events,
    temporal_phase_shift_temporal_frame_video,
)
from model.temporal_memory_net import BidirectionalTemporalMemoryNet


def _as_bool(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {'true', '1', 'yes', 'on'}:
            return True
        if value in {'false', '0', 'no', 'off'}:
            return False
        raise ValueError('Expected a boolean value, got {!r}.'.format(value))
    return bool(value)


@dataclass(frozen=True)
class TemporalMemoryInferenceConfig:
    """Configuration for a full-stream bidirectional temporal-memory expert."""

    enabled: bool = False
    model_path: str = ''
    sparse_weight: float = 0.5
    secondary_model_path: str = ''
    primary_weight: float = 1.0
    secondary_max_event_count: int = 0
    blend_model_path: str = ''
    blend_primary_weight: float = 1.0
    dense_specialist_enabled: bool = False
    dense_specialist_model_path: str = ''
    dense_specialist_event_count_cutoff: int = 100000
    dense_specialist_weight: float = 0.5
    fine_time_expert_enabled: bool = False
    fine_time_expert_model_path: str = ''
    fine_time_expert_event_count_cutoff: int = 30000
    fine_time_expert_weight: float = 0.25
    fine_time_expert_bin_size: int = 25
    fine_time_expert_context_bins: int = 5
    fine_time_expert_sequence_length: int = 32
    phase_specialist_enabled: bool = False
    phase_specialist_model_path: str = ''
    phase_specialist_event_count_cutoff: int = 30000
    phase_specialist_weight: float = 0.25
    phase_specialist_offset: int = 25
    phase_specialist_blend_compatible: bool = False
    local_temporal_context_enabled: bool = False
    local_temporal_context_kernel_size: int = 11

    def __post_init__(self):
        if self.enabled and not self.model_path:
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_model_path is required when '
                'temporal memory is enabled.'
            )
        if not 0.0 <= self.primary_weight <= 1.0:
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_primary_weight must be in [0, 1].'
            )
        if not 0.0 <= self.blend_primary_weight <= 1.0:
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_blend_primary_weight '
                'must be in [0, 1].'
            )
        if self.dense_specialist_enabled and not self.dense_specialist_model_path:
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_dense_specialist_model_path '
                'is required when the dense specialist is enabled.'
            )
        if (
            self.dense_specialist_enabled
            and self.dense_specialist_event_count_cutoff <= 0
        ):
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_dense_specialist_event_count_cutoff '
                'must be positive when the dense specialist is enabled.'
            )
        if not 0.0 <= self.dense_specialist_weight <= 1.0:
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_dense_specialist_weight '
                'must be in [0, 1].'
            )
        if self.fine_time_expert_enabled and not self.enabled:
            raise ValueError(
                'TEMPORAL_MEMORY temporal_memory_enabled is required when the '
                'fine-time expert is enabled.'
            )
        if self.fine_time_expert_enabled and not self.fine_time_expert_model_path:
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_fine_time_expert_model_path '
                'is required when the fine-time expert is enabled.'
            )
        if (
            self.fine_time_expert_enabled
            and self.fine_time_expert_event_count_cutoff <= 0
        ):
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_fine_time_expert_event_count_cutoff '
                'must be positive when the fine-time expert is enabled.'
            )
        if not 0.0 <= self.fine_time_expert_weight <= 1.0:
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_fine_time_expert_weight '
                'must be in [0, 1].'
            )
        if self.fine_time_expert_bin_size <= 0:
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_fine_time_expert_bin_size '
                'must be positive.'
            )
        if (
            self.fine_time_expert_context_bins <= 0
            or self.fine_time_expert_context_bins % 2 == 0
        ):
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_fine_time_expert_context_bins '
                'must be a positive odd integer.'
            )
        if self.fine_time_expert_sequence_length <= 1:
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_fine_time_expert_sequence_length '
                'must exceed one.'
            )
        if self.phase_specialist_enabled and not self.enabled:
            raise ValueError(
                'TEMPORAL_MEMORY temporal_memory_enabled is required when the '
                'phase specialist is enabled.'
            )
        if self.phase_specialist_enabled and not self.phase_specialist_model_path:
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_phase_specialist_model_path '
                'is required when the phase specialist is enabled.'
            )
        if (
            self.phase_specialist_enabled
            and self.phase_specialist_event_count_cutoff <= 0
        ):
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_phase_specialist_event_count_cutoff '
                'must be positive when the phase specialist is enabled.'
            )
        if not 0.0 <= self.phase_specialist_weight <= 1.0:
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_phase_specialist_weight '
                'must be in [0, 1].'
            )
        if self.phase_specialist_offset <= 0:
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_phase_specialist_offset '
                'must be positive.'
            )
        if self.local_temporal_context_kernel_size <= 0 or (
            self.local_temporal_context_kernel_size % 2 == 0
        ):
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_local_temporal_context_kernel_size '
                'must be a positive odd integer.'
            )
        if self.secondary_max_event_count < 0:
            raise ValueError(
                'TEMPORAL_MEMORY.temporal_memory_secondary_max_event_count '
                'must be non-negative.'
            )

    @property
    def memory_only(self):
        return self.enabled and self.sparse_weight == 0.0

    @property
    def has_secondary_model(self):
        return bool(self.secondary_model_path.strip())

    @property
    def has_blend_model(self):
        return bool(self.blend_model_path.strip())

    @property
    def routes_secondary_by_event_count(self):
        return self.has_secondary_model and self.secondary_max_event_count > 0

    def use_secondary_for_event_count(self, event_count):
        return (
            self.routes_secondary_by_event_count
            and int(event_count) <= self.secondary_max_event_count
        )

    def use_dense_specialist_for_event_count(self, event_count):
        return (
            self.dense_specialist_enabled
            and int(event_count) > self.dense_specialist_event_count_cutoff
        )

    def use_fine_time_expert_for_event_count(self, event_count):
        return (
            self.fine_time_expert_enabled
            and int(event_count) > self.fine_time_expert_event_count_cutoff
        )

    def use_phase_specialist_for_event_count(self, event_count):
        return (
            self.phase_specialist_enabled
            and int(event_count) > self.phase_specialist_event_count_cutoff
        )

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(getattr(cfg, 'temporal_memory_enabled', False)),
            model_path=str(getattr(cfg, 'temporal_memory_model_path', '')),
            sparse_weight=float(getattr(cfg, 'temporal_memory_sparse_weight', 0.5)),
            secondary_model_path=str(
                getattr(cfg, 'temporal_memory_secondary_model_path', '')
            ),
            primary_weight=float(
                getattr(cfg, 'temporal_memory_primary_weight', 1.0)
            ),
            secondary_max_event_count=int(
                getattr(cfg, 'temporal_memory_secondary_max_event_count', 0)
            ),
            blend_model_path=str(
                getattr(cfg, 'temporal_memory_blend_model_path', '')
            ),
            blend_primary_weight=float(
                getattr(cfg, 'temporal_memory_blend_primary_weight', 1.0)
            ),
            dense_specialist_enabled=_as_bool(
                getattr(
                    cfg,
                    'temporal_memory_dense_specialist_enabled',
                    False,
                )
            ),
            dense_specialist_model_path=str(
                getattr(
                    cfg,
                    'temporal_memory_dense_specialist_model_path',
                    '',
                )
            ),
            dense_specialist_event_count_cutoff=int(
                getattr(
                    cfg,
                    'temporal_memory_dense_specialist_event_count_cutoff',
                    100000,
                )
            ),
            dense_specialist_weight=float(
                getattr(cfg, 'temporal_memory_dense_specialist_weight', 0.5)
            ),
            fine_time_expert_enabled=_as_bool(
                getattr(cfg, 'temporal_memory_fine_time_expert_enabled', False)
            ),
            fine_time_expert_model_path=str(
                getattr(cfg, 'temporal_memory_fine_time_expert_model_path', '')
            ),
            fine_time_expert_event_count_cutoff=int(
                getattr(
                    cfg,
                    'temporal_memory_fine_time_expert_event_count_cutoff',
                    30000,
                )
            ),
            fine_time_expert_weight=float(
                getattr(cfg, 'temporal_memory_fine_time_expert_weight', 0.25)
            ),
            fine_time_expert_bin_size=int(
                getattr(cfg, 'temporal_memory_fine_time_expert_bin_size', 25)
            ),
            fine_time_expert_context_bins=int(
                getattr(cfg, 'temporal_memory_fine_time_expert_context_bins', 5)
            ),
            fine_time_expert_sequence_length=int(
                getattr(cfg, 'temporal_memory_fine_time_expert_sequence_length', 32)
            ),
            phase_specialist_enabled=_as_bool(
                getattr(cfg, 'temporal_memory_phase_specialist_enabled', False)
            ),
            phase_specialist_model_path=str(
                getattr(cfg, 'temporal_memory_phase_specialist_model_path', '')
            ),
            phase_specialist_event_count_cutoff=int(
                getattr(
                    cfg,
                    'temporal_memory_phase_specialist_event_count_cutoff',
                    30000,
                )
            ),
            phase_specialist_weight=float(
                getattr(cfg, 'temporal_memory_phase_specialist_weight', 0.25)
            ),
            phase_specialist_offset=int(
                getattr(cfg, 'temporal_memory_phase_specialist_offset', 25)
            ),
            phase_specialist_blend_compatible=_as_bool(
                getattr(
                    cfg,
                    'temporal_memory_phase_specialist_blend_compatible',
                    False,
                )
            ),
            local_temporal_context_enabled=_as_bool(
                getattr(
                    cfg,
                    'temporal_memory_local_temporal_context_enabled',
                    False,
                )
            ),
            local_temporal_context_kernel_size=int(
                getattr(
                    cfg,
                    'temporal_memory_local_temporal_context_kernel_size',
                    11,
                )
            ),
        )

    def describe(self):
        if not self.enabled:
            return 'disabled'
        description = (
            'enabled (sparse_weight={:.3f}, memory_weight={:.3f}, model={}'
        ).format(
            self.sparse_weight, 1.0 - self.sparse_weight, self.model_path
        )
        if self.has_secondary_model:
            description += ', secondary_model={}'.format(self.secondary_model_path)
            if self.routes_secondary_by_event_count:
                description += ', secondary for event_count <= {}'.format(
                    self.secondary_max_event_count
                )
            else:
                description += ', primary_weight={:.3f}'.format(self.primary_weight)
        if self.has_blend_model:
            description += ', high-density blend_model={}'.format(
                self.blend_model_path
            )
            description += ', blend_primary_weight={:.3f}'.format(
                self.blend_primary_weight
            )
        if self.dense_specialist_enabled:
            description += (
                ', dense_specialist={} for event_count > {}, weight={:.3f}'
            ).format(
                self.dense_specialist_model_path,
                self.dense_specialist_event_count_cutoff,
                self.dense_specialist_weight,
            )
        if self.fine_time_expert_enabled:
            description += (
                ', fine_time_expert={} for event_count > {}, weight={:.3f}, '
                'bin_size={}, context_bins={}, sequence_length={}'
            ).format(
                self.fine_time_expert_model_path,
                self.fine_time_expert_event_count_cutoff,
                self.fine_time_expert_weight,
                self.fine_time_expert_bin_size,
                self.fine_time_expert_context_bins,
                self.fine_time_expert_sequence_length,
            )
        if self.phase_specialist_enabled:
            description += (
                ', phase_specialist={} for event_count > {}, weight={:.3f}, '
                'offset={}'
            ).format(
                self.phase_specialist_model_path,
                self.phase_specialist_event_count_cutoff,
                self.phase_specialist_weight,
                self.phase_specialist_offset,
            )
        if self.local_temporal_context_enabled:
            description += ', local_temporal_context_kernel={}'.format(
                self.local_temporal_context_kernel_size
            )
        return description + ')'


@dataclass(frozen=True)
class TemporalPhaseTTAConfig:
    """Optional half-bin temporal re-quantization before score averaging."""

    enabled: bool = False
    phase_offset: int = 25
    original_weight: float = 1.0
    min_event_count: int = 0
    average_mode: str = 'probability'
    boundary_adaptive: bool = False

    def __post_init__(self):
        if self.enabled and self.phase_offset <= 0:
            raise ValueError('Temporal phase offset must be positive when enabled.')
        if not 0.0 <= self.original_weight <= 1.0:
            raise ValueError('Temporal phase original weight must be in [0, 1].')
        if self.min_event_count < 0:
            raise ValueError('Temporal phase minimum event count must be non-negative.')
        if self.average_mode not in {'probability', 'logit'}:
            raise ValueError(
                "Temporal phase average_mode must be 'probability' or 'logit'."
            )

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(
                getattr(cfg, 'p41_temporal_phase_enabled', False)
            ),
            phase_offset=int(
                getattr(cfg, 'p41_temporal_phase_offset', 25)
            ),
            original_weight=float(
                getattr(cfg, 'p41_temporal_phase_original_weight', 1.0)
            ),
            min_event_count=int(
                getattr(cfg, 'p41_temporal_phase_min_event_count', 0)
            ),
            average_mode=str(
                getattr(cfg, 'p41_temporal_phase_average_mode', 'probability')
            ),
            boundary_adaptive=_as_bool(
                getattr(cfg, 'p41_temporal_phase_boundary_adaptive', False)
            ),
        )

    def describe(self):
        if not self.enabled:
            return 'disabled'
        return ('enabled (offset={}, original_weight={:.3f}, min_event_count={}, '
                'average_mode={}, boundary_adaptive={})').format(
            self.phase_offset,
            self.original_weight,
            self.min_event_count,
            self.average_mode,
            self.boundary_adaptive,
        )


@dataclass(frozen=True)
class TemporalReverseTTAConfig:
    """Optionally average a bidirectional-memory prediction from reversed time."""

    enabled: bool = False
    original_weight: float = 1.0
    min_event_count: int = 0

    def __post_init__(self):
        if not 0.0 <= self.original_weight <= 1.0:
            raise ValueError('Temporal reverse original weight must be in [0, 1].')
        if self.min_event_count < 0:
            raise ValueError('Temporal reverse minimum event count must be non-negative.')

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(
                getattr(cfg, 'p54_temporal_reverse_enabled', False)
            ),
            original_weight=float(
                getattr(cfg, 'p54_temporal_reverse_original_weight', 1.0)
            ),
            min_event_count=int(
                getattr(cfg, 'p54_temporal_reverse_min_event_count', 0)
            ),
        )

    def describe(self):
        if not self.enabled:
            return 'disabled'
        return 'enabled (original_weight={:.3f}, min_event_count={})'.format(
            self.original_weight,
            self.min_event_count,
        )


@dataclass(frozen=True)
class TemporalMultiPhaseTTAConfig:
    """Optional evenly weighted multi-phase temporal cycle-spinning."""

    enabled: bool = False
    phase_offsets: tuple = (12, 25, 37)
    original_weight: float = 1.0
    min_event_count: int = 0

    def __post_init__(self):
        offsets = tuple(int(offset) for offset in self.phase_offsets)
        if self.enabled and not offsets:
            raise ValueError('Temporal multi-phase offsets are required when enabled.')
        if any(offset <= 0 for offset in offsets):
            raise ValueError('Temporal multi-phase offsets must be positive.')
        if len(set(offsets)) != len(offsets):
            raise ValueError('Temporal multi-phase offsets must be unique.')
        if not 0.0 <= self.original_weight <= 1.0:
            raise ValueError('Temporal multi-phase original weight must be in [0, 1].')
        if self.min_event_count < 0:
            raise ValueError('Temporal multi-phase minimum event count must be non-negative.')
        object.__setattr__(self, 'phase_offsets', offsets)

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(
                getattr(cfg, 'p50_temporal_multiphase_enabled', False)
            ),
            phase_offsets=tuple(
                getattr(cfg, 'p50_temporal_multiphase_offsets', [12, 25, 37])
            ),
            original_weight=float(
                getattr(cfg, 'p50_temporal_multiphase_original_weight', 1.0)
            ),
            min_event_count=int(
                getattr(cfg, 'p50_temporal_multiphase_min_event_count', 0)
            ),
        )

    def describe(self):
        if not self.enabled:
            return 'disabled'
        return 'enabled (offsets={}, original_weight={:.3f}, min_event_count={})'.format(
            self.phase_offsets,
            self.original_weight,
            self.min_event_count,
        )


@dataclass(frozen=True)
class SpatialPhaseTTAConfig:
    """Optional 2x2 spatial cycle-spinning for full-stream memory scores."""

    enabled: bool = False
    offset: int = 1
    min_event_count: int = 0

    def __post_init__(self):
        if self.enabled and self.offset <= 0:
            raise ValueError('Spatial phase offset must be positive when enabled.')
        if self.min_event_count < 0:
            raise ValueError('Spatial phase minimum event count must be non-negative.')

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(
                getattr(cfg, 'p44_spatial_phase_enabled', False)
            ),
            offset=int(getattr(cfg, 'p44_spatial_phase_offset', 1)),
            min_event_count=int(
                getattr(cfg, 'p44_spatial_phase_min_event_count', 0)
            ),
        )

    def describe(self):
        if not self.enabled:
            return 'disabled'
        return 'enabled (2x2 cycle, offset={}, min_event_count={})'.format(
            self.offset,
            self.min_event_count,
        )


def load_temporal_memory_model(
    checkpoint_path,
    device,
    context_bins,
    width,
    sequence_length=None,
):
    """Load a temporal-memory checkpoint and validate its saved architecture.

    ``width=None`` is used for auxiliary routed experts.  Their channel width
    is taken from checkpoint metadata so a widened primary model can coexist
    with the historical width-16 M10/M111 experts without changing either
    model's weights or outputs.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            'Temporal-memory checkpoint not found: {}'.format(checkpoint_path)
        )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved = checkpoint.get('temporal_memory', {})
    saved_context_bins = saved.get('context_bins')
    saved_width = saved.get('width')
    saved_normalization_max_groups = int(
        saved.get('normalization_max_groups', 8)
    )
    saved_sequence_length = saved.get('sequence_length')
    saved_density_calibration = bool(
        saved.get('density_calibration_enabled', False)
    )
    saved_confidence_head = bool(
        saved.get('confidence_head_enabled', False)
    )
    saved_temporal_attention = bool(
        saved.get('temporal_attention_enabled', False)
    )
    saved_temporal_diff = bool(saved.get('temporal_diff_enabled', False))
    saved_ssa = bool(saved.get('ssa_enabled', False))
    saved_temporal_attention_num_heads = int(
        saved.get('temporal_attention_num_heads', 4)
    )
    saved_attention_relative_bias = bool(
        saved.get('attention_relative_bias_enabled', False)
    )
    saved_attention_relative_bias_max_distance = int(
        saved.get('attention_relative_bias_max_distance', 8)
    )
    saved_advection_alignment = bool(
        saved.get('advection_alignment_enabled', False)
    )
    saved_advection_max_flow = float(saved.get('advection_max_flow', 0.0))
    saved_fine_temporal_memory = bool(
        saved.get('fine_temporal_memory_enabled', False)
    )
    saved_fine_advection_max_flow = float(
        saved.get('fine_advection_max_flow', 0.0)
    )
    saved_target_center = bool(saved.get('target_center_enabled', False))
    saved_target_level = bool(saved.get('target_level_enabled', False))
    saved_target_level_downsample = int(
        saved.get('target_level_downsample', 4)
    )
    saved_objectness_gate = bool(
        saved.get('objectness_gate_enabled', False)
    )
    saved_objectness_gate_strength = float(
        saved.get('objectness_gate_strength', 0.5)
    )
    saved_objectness_gate_downsample = int(
        saved.get('objectness_gate_downsample', 4)
    )
    saved_center_memory = bool(saved.get('center_memory_enabled', False))
    saved_center_memory_channels = int(
        saved.get('center_memory_channels', 4)
    )
    saved_center_memory_downsample = int(
        saved.get('center_memory_downsample', 4)
    )
    saved_local_temporal_context = bool(
        saved.get('local_temporal_context_enabled', False)
    )
    saved_local_temporal_context_kernel_size = int(
        saved.get('local_temporal_context_kernel_size', 11)
    )
    if saved_context_bins is not None and int(saved_context_bins) != int(context_bins):
        raise ValueError(
            'Checkpoint context_bins={} does not match configured {}.'.format(
                saved_context_bins, context_bins
            )
        )
    model_width = saved_width if width is None else width
    if model_width is None:
        raise ValueError(
            'Checkpoint does not contain width metadata and no width was configured.'
        )
    if saved_width is not None and width is not None and int(saved_width) != int(width):
        raise ValueError(
            'Checkpoint width={} does not match configured {}.'.format(
                saved_width, width
            )
        )
    if (
        sequence_length is not None
        and saved_sequence_length is not None
        and int(saved_sequence_length) != int(sequence_length)
    ):
        raise ValueError(
            'Checkpoint sequence_length={} does not match configured {}.'.format(
                saved_sequence_length, sequence_length
            )
        )
    model = BidirectionalTemporalMemoryNet(
        input_channels=int(context_bins) * 2,
        width=int(model_width),
        normalization_max_groups=saved_normalization_max_groups,
        density_calibration_enabled=saved_density_calibration,
        confidence_head_enabled=saved_confidence_head,
        temporal_diff_enabled=saved_temporal_diff,
        ssa_enabled=saved_ssa,
        temporal_attention_enabled=saved_temporal_attention,
        temporal_attention_num_heads=saved_temporal_attention_num_heads,
        temporal_attention_relative_bias_enabled=saved_attention_relative_bias,
        temporal_attention_relative_bias_max_distance=(
            saved_attention_relative_bias_max_distance
        ),
        advection_alignment_enabled=saved_advection_alignment,
        advection_max_flow=saved_advection_max_flow,
        fine_temporal_memory_enabled=saved_fine_temporal_memory,
        fine_advection_max_flow=saved_fine_advection_max_flow,
        local_temporal_context_enabled=saved_local_temporal_context,
        local_temporal_context_kernel_size=(
            saved_local_temporal_context_kernel_size
        ),
        target_center_enabled=saved_target_center,
        target_level_enabled=saved_target_level,
        target_level_downsample=saved_target_level_downsample,
        objectness_gate_enabled=saved_objectness_gate,
        objectness_gate_strength=saved_objectness_gate_strength,
        objectness_gate_downsample=saved_objectness_gate_downsample,
        center_memory_enabled=saved_center_memory,
        center_memory_channels=saved_center_memory_channels,
        center_memory_downsample=saved_center_memory_downsample,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model, checkpoint


def _frame_tensor(
    video,
    temporal_bins,
    context_bins,
    width,
    height,
    log_count_clip,
    device,
    local_temporal_context_enabled=False,
    local_temporal_context_kernel_size=11,
):
    frames = np.stack(
        [
            build_temporal_context_frame(
                video,
                temporal_bin,
                context_bins,
                width,
                height,
                log_count_clip,
                local_temporal_context_enabled=local_temporal_context_enabled,
                local_temporal_context_kernel_size=(
                    local_temporal_context_kernel_size
                ),
            )
            for temporal_bin in temporal_bins
        ],
        axis=0,
    )
    return torch.from_numpy(frames).float().to(device)


def horizontal_flip_temporal_memory_video(video, width):
    """Return an event-order-preserving horizontal mirror of ``video``.

    Time bins, labels, and target ids stay aligned with the original event
    indices. Therefore prediction scores from the mirrored stream already
    correspond to the original event order and can be averaged directly.
    """
    width = int(width)
    if width <= 0:
        raise ValueError('width must be positive.')
    locations = np.asarray(video.locations)
    if locations.ndim != 2 or locations.shape[1] < 3:
        raise ValueError('video.locations must have shape [N, 3+].')
    if locations.size and (locations[:, 0].min() < 0 or locations[:, 0].max() >= width):
        raise ValueError('video x coordinates are outside the configured image width.')
    mirrored_locations = locations.copy()
    mirrored_locations[:, 0] = width - 1 - mirrored_locations[:, 0]
    return replace(video, locations=mirrored_locations)


def polarity_invert_temporal_memory_video(video):
    """Swap OFF/ON event channels while preserving event index alignment."""
    polarities = np.asarray(video.polarities)
    if polarities.ndim != 1 or polarities.shape[0] != video.locations.shape[0]:
        raise ValueError('video polarities must be flat and align with event locations.')
    if polarities.size and not np.all((polarities == 0.0) | (polarities == 1.0)):
        raise ValueError('polarity inversion requires binary 0/1 polarities.')
    return replace(
        video,
        polarities=(1.0 - polarities).astype(np.float32, copy=False),
    )


def temporal_phase_shift_temporal_memory_video(
    video,
    temporal_bin_size,
    phase_offset,
):
    """Re-bin a stream after a positive temporal phase shift.

    This transformation is only used to build an alternate inference view.
    Event order is preserved, so the returned score vector remains aligned to
    the original event array and its original coordinates can still be passed
    to P6/P0/P18 unchanged.
    """
    return temporal_phase_shift_temporal_frame_video(
        video,
        temporal_bin_size,
        phase_offset,
    )


def temporal_reverse_temporal_memory_video(video, whole_t):
    """Reverse metric time while preserving event-score index alignment.

    The temporal-memory network has independent forward and backward ConvGRUs.
    Reversing a complete event stream exchanges their directional roles without
    modifying coordinates, polarity, labels, or original event ordering.  The
    returned score array consequently remains aligned to the unmodified event
    indices and can be blended before P6/P0/P18.
    """
    whole_t = int(whole_t)
    if whole_t <= 0:
        raise ValueError('whole_t must be positive.')
    locations = np.asarray(video.locations)
    if locations.ndim != 2 or locations.shape[1] < 3:
        raise ValueError('video.locations must have shape [N, 3+].')
    if locations.size and (
        locations[:, 2].min() < 0 or locations[:, 2].max() >= whole_t
    ):
        raise ValueError('video timestamps are outside [0, whole_t).')
    reversed_locations = locations.copy()
    reversed_locations[:, 2] = whole_t - 1 - reversed_locations[:, 2]
    return temporal_frame_video_from_events(
        name=video.name,
        locations=reversed_locations,
        polarities=video.polarities,
        temporal_bin_size=(
            whole_t // len(video.event_indices_by_bin)
            if whole_t % len(video.event_indices_by_bin) == 0
            else None
        ),
        whole_t=whole_t,
        labels=video.labels,
        target_ids=video.target_ids,
    )


def spatial_phase_shift_temporal_memory_video(
    video,
    width,
    height,
    x_offset,
    y_offset,
):
    """Cycle spatial event coordinates while preserving event-score alignment.

    A four-phase ensemble removes stride-induced pixel-grid preference without
    dropping border events. Circular remapping affects only frame formation;
    returned scores still index the original event order and coordinates.
    """
    width = int(width)
    height = int(height)
    x_offset = int(x_offset)
    y_offset = int(y_offset)
    if width <= 0 or height <= 0:
        raise ValueError('width and height must be positive.')
    if x_offset < 0 or x_offset >= width:
        raise ValueError('x_offset must be in [0, width - 1].')
    if y_offset < 0 or y_offset >= height:
        raise ValueError('y_offset must be in [0, height - 1].')
    if x_offset == 0 and y_offset == 0:
        raise ValueError('At least one spatial phase offset must be positive.')
    locations = np.asarray(video.locations)
    if locations.ndim != 2 or locations.shape[1] < 3:
        raise ValueError('video.locations must have shape [N, 3+].')
    if locations.size and (
        locations[:, 0].min() < 0
        or locations[:, 0].max() >= width
        or locations[:, 1].min() < 0
        or locations[:, 1].max() >= height
    ):
        raise ValueError('video coordinates are outside the configured image bounds.')
    shifted_locations = locations.copy()
    shifted_locations[:, 0] = (shifted_locations[:, 0] + x_offset) % width
    shifted_locations[:, 1] = (shifted_locations[:, 1] + y_offset) % height
    return replace(video, locations=shifted_locations)


def predict_temporal_memory_scores(
    model,
    video,
    device,
    context_bins,
    width,
    height,
    inference_batch_size,
    log_count_clip=4.0,
    local_temporal_context_enabled=None,
    local_temporal_context_kernel_size=None,
):
    """Return one probability per event using bidirectional full-stream memory.

    Bottleneck maps are kept for all temporal bins, while skip features are
    recomputed in a second pass. This keeps inference within the 4GB budget.
    """
    context_bins = int(context_bins)
    width = int(width)
    height = int(height)
    inference_batch_size = int(inference_batch_size)
    if context_bins < 1 or context_bins % 2 == 0:
        raise ValueError('context_bins must be a positive odd integer.')
    if inference_batch_size <= 0:
        raise ValueError('inference_batch_size must be positive.')
    model_local_temporal_context = bool(
        getattr(model, 'local_temporal_context_enabled', False)
    )
    if local_temporal_context_enabled is None:
        local_temporal_context_enabled = model_local_temporal_context
    if bool(local_temporal_context_enabled) != model_local_temporal_context:
        raise ValueError(
            'Local temporal-context input setting does not match checkpoint model.'
        )
    if local_temporal_context_kernel_size is None:
        local_temporal_context_kernel_size = int(
            getattr(model, 'local_temporal_context_kernel_size', 11)
        )
    local_temporal_context_kernel_size = int(local_temporal_context_kernel_size)
    temporal_bin_count = len(video.event_indices_by_bin)
    if temporal_bin_count <= 0:
        raise ValueError('video must contain temporal bins.')

    bottlenecks = []
    fine_level2_features = []
    fine_memory_enabled = bool(
        getattr(model, 'fine_temporal_memory_enabled', False)
    )
    center_memory_enabled = bool(
        getattr(model, 'center_memory_enabled', False)
    )
    with torch.no_grad():
        for start in range(0, temporal_bin_count, inference_batch_size):
            temporal_bins = list(
                range(start, min(start + inference_batch_size, temporal_bin_count))
            )
            frames = _frame_tensor(
                video,
                temporal_bins,
                context_bins,
                width,
                height,
                log_count_clip,
                device,
                local_temporal_context_enabled=model_local_temporal_context,
                local_temporal_context_kernel_size=local_temporal_context_kernel_size,
            )
            if fine_memory_enabled:
                level2, bottleneck = model.encode_memory_features(frames)
                fine_level2_features.append(level2)
                bottlenecks.append(bottleneck)
            else:
                bottlenecks.append(model.encode_bottleneck(frames))
        residuals = model.temporal_residual(torch.cat(bottlenecks, dim=0))
        fine_residuals = None
        if fine_memory_enabled:
            fine_residuals = model.fine_temporal_residual(
                torch.cat(fine_level2_features, dim=0)
            )
        center_residuals = None
        if center_memory_enabled:
            center_logits = []
            for start in range(0, temporal_bin_count, inference_batch_size):
                temporal_bins = list(
                    range(start, min(start + inference_batch_size, temporal_bin_count))
                )
                frames = _frame_tensor(
                    video,
                    temporal_bins,
                    context_bins,
                    width,
                    height,
                    log_count_clip,
                    device,
                    local_temporal_context_enabled=model_local_temporal_context,
                    local_temporal_context_kernel_size=(
                        local_temporal_context_kernel_size
                    ),
                )
                _, batch_center_logits = model.decode_with_residual(
                    frames,
                    residuals[start:start + len(temporal_bins)],
                    fine_residual=(
                        fine_residuals[start:start + len(temporal_bins)]
                        if fine_residuals is not None
                        else None
                    ),
                    return_target_center_logits=True,
                )
                center_logits.append(batch_center_logits)
            center_residuals = model.center_temporal_residual(
                torch.cat(center_logits, dim=0)
            )

    scores = np.empty(video.locations.shape[0], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, temporal_bin_count, inference_batch_size):
            temporal_bins = list(
                range(start, min(start + inference_batch_size, temporal_bin_count))
            )
            frames = _frame_tensor(
                video,
                temporal_bins,
                context_bins,
                width,
                height,
                log_count_clip,
                device,
                local_temporal_context_enabled=model_local_temporal_context,
                local_temporal_context_kernel_size=local_temporal_context_kernel_size,
            )
            confidence_enabled = bool(
                getattr(model, 'confidence_head_enabled', False)
            )
            decoded = model.decode_with_residual(
                frames,
                residuals[start:start + len(temporal_bins)],
                fine_residual=(
                    fine_residuals[start:start + len(temporal_bins)]
                    if fine_residuals is not None
                    else None
                ),
                return_confidence_logits=confidence_enabled,
                center_memory_residual=(
                    center_residuals[start:start + len(temporal_bins)]
                    if center_residuals is not None
                    else None
                ),
            )
            if confidence_enabled:
                logit_maps, confidence_maps = decoded
                probabilities = torch.sigmoid(
                    logit_maps + confidence_maps
                ).squeeze(1).cpu().numpy()
            else:
                probabilities = torch.sigmoid(
                    decoded
                ).squeeze(1).cpu().numpy()
            for local_index, temporal_bin in enumerate(temporal_bins):
                event_indices = video.event_indices_by_bin[temporal_bin]
                if event_indices.size == 0:
                    continue
                locations = video.locations[event_indices]
                event_probabilities = probabilities[
                    local_index,
                    locations[:, 1],
                    locations[:, 0],
                ]
                scores[event_indices] = event_probabilities
    if not np.isfinite(scores).all():
        raise RuntimeError('Temporal-memory inference produced non-finite scores.')
    return torch.from_numpy(scores)


def predict_temporal_memory_scores_with_horizontal_flip(
    model,
    video,
    device,
    context_bins,
    width,
    height,
    inference_batch_size,
    log_count_clip=4.0,
    horizontal_flip_enabled=False,
    original_weight=1.0,
):
    """Optionally average full-stream scores from original and mirrored events."""
    original_weight = float(original_weight)
    if not 0.0 <= original_weight <= 1.0:
        raise ValueError('original_weight must be in [0, 1].')
    original_scores = predict_temporal_memory_scores(
        model,
        video,
        device,
        context_bins,
        width,
        height,
        inference_batch_size,
        log_count_clip,
    )
    if not horizontal_flip_enabled or original_weight >= 1.0:
        return original_scores
    mirrored_scores = predict_temporal_memory_scores(
        model,
        horizontal_flip_temporal_memory_video(video, width),
        device,
        context_bins,
        width,
        height,
        inference_batch_size,
        log_count_clip,
    )
    return (
        original_scores * original_weight
        + mirrored_scores * (1.0 - original_weight)
    )


def predict_temporal_memory_scores_with_temporal_phase(
    model,
    video,
    device,
    context_bins,
    width,
    height,
    inference_batch_size,
    temporal_bin_size,
    phase_offset,
    log_count_clip=4.0,
    temporal_phase_enabled=False,
    original_weight=1.0,
    average_mode='probability',
    boundary_adaptive=False,
):
    """Optionally average original scores with a phase-shifted time grid."""
    original_weight = float(original_weight)
    if not 0.0 <= original_weight <= 1.0:
        raise ValueError('original_weight must be in [0, 1].')
    original_scores = predict_temporal_memory_scores(
        model,
        video,
        device,
        context_bins,
        width,
        height,
        inference_batch_size,
        log_count_clip,
    )
    if not temporal_phase_enabled or original_weight >= 1.0:
        return original_scores
    phase_scores = predict_temporal_memory_scores(
        model,
        temporal_phase_shift_temporal_memory_video(
            video,
            temporal_bin_size,
            phase_offset,
        ),
        device,
        context_bins,
        width,
        height,
        inference_batch_size,
        log_count_clip,
    )
    if boundary_adaptive:
        return blend_temporal_phase_scores_boundary_adaptive(
            original_scores,
            phase_scores,
            video.locations[:, 2],
            temporal_bin_size,
            phase_offset,
        )
    return blend_temporal_phase_scores(
        original_scores,
        phase_scores,
        original_weight,
        average_mode,
    )


def predict_temporal_memory_scores_with_temporal_multiphase(
    model,
    video,
    device,
    context_bins,
    width,
    height,
    inference_batch_size,
    temporal_bin_size,
    phase_offsets,
    log_count_clip=4.0,
    temporal_multiphase_enabled=False,
    original_weight=1.0,
):
    """Average one raw and several evenly weighted shifted temporal views."""
    phase_offsets = tuple(int(offset) for offset in phase_offsets)
    original_weight = float(original_weight)
    if not 0.0 <= original_weight <= 1.0:
        raise ValueError('original_weight must be in [0, 1].')
    if not temporal_multiphase_enabled or original_weight >= 1.0:
        return predict_temporal_memory_scores(
            model,
            video,
            device,
            context_bins,
            width,
            height,
            inference_batch_size,
            log_count_clip,
        )
    if not phase_offsets:
        raise ValueError('phase_offsets must not be empty when multi-phase TTA is enabled.')
    if len(set(phase_offsets)) != len(phase_offsets):
        raise ValueError('phase_offsets must be unique.')
    if any(offset <= 0 or offset >= temporal_bin_size for offset in phase_offsets):
        raise ValueError('phase_offsets must be inside the temporal bin.')
    scores = predict_temporal_memory_scores(
        model,
        video,
        device,
        context_bins,
        width,
        height,
        inference_batch_size,
        log_count_clip,
    ) * original_weight
    phase_weight = (1.0 - original_weight) / len(phase_offsets)
    for phase_offset in phase_offsets:
        scores = scores + predict_temporal_memory_scores(
            model,
            temporal_phase_shift_temporal_memory_video(
                video,
                temporal_bin_size,
                phase_offset,
            ),
            device,
            context_bins,
            width,
            height,
            inference_batch_size,
            log_count_clip,
        ) * phase_weight
    return scores


def blend_temporal_phase_scores(
    original_scores,
    phase_scores,
    original_weight,
    average_mode='probability',
):
    """Combine aligned temporal views in probability or log-odds space."""
    if original_scores.shape != phase_scores.shape:
        raise ValueError(
            'Temporal phase score shapes do not match: {} vs {}.'.format(
                tuple(original_scores.shape), tuple(phase_scores.shape)
            )
        )
    original_weight = float(original_weight)
    if not 0.0 <= original_weight <= 1.0:
        raise ValueError('original_weight must be in [0, 1].')
    if average_mode == 'probability':
        return (
            original_scores * original_weight
            + phase_scores * (1.0 - original_weight)
        )
    if average_mode != 'logit':
        raise ValueError("average_mode must be 'probability' or 'logit'.")
    epsilon = torch.finfo(original_scores.dtype).eps
    original_logits = torch.logit(
        original_scores.clamp(min=epsilon, max=1.0 - epsilon)
    )
    phase_logits = torch.logit(
        phase_scores.clamp(min=epsilon, max=1.0 - epsilon)
    )
    return torch.sigmoid(
        original_logits * original_weight
        + phase_logits * (1.0 - original_weight)
    )


def blend_temporal_phase_scores_boundary_adaptive(
    original_scores,
    phase_scores,
    timestamps,
    temporal_bin_size,
    phase_offset,
):
    """Prefer the shifted view near raw-bin boundaries without tuning a weight.

    A half-bin phase shift is most informative where a raw 50-unit bin cuts a
    trajectory.  For each event, use an original-view weight from 0.5 at a
    raw-bin boundary to 1.0 at the bin centre.  Under uniform time coverage
    its mean is exactly 0.75, preserving P41's established aggregate balance
    while making the alternate view local to the discretization error it can
    correct.
    """
    if original_scores.shape != phase_scores.shape:
        raise ValueError(
            'Temporal phase score shapes do not match: {} vs {}.'.format(
                tuple(original_scores.shape), tuple(phase_scores.shape)
            )
        )
    temporal_bin_size = int(temporal_bin_size)
    phase_offset = int(phase_offset)
    if temporal_bin_size <= 1:
        raise ValueError('temporal_bin_size must exceed one.')
    if phase_offset * 2 != temporal_bin_size:
        raise ValueError(
            'Boundary-adaptive phase blending requires a half-bin phase offset.'
        )
    timestamps = np.asarray(timestamps)
    if timestamps.ndim != 1 or timestamps.shape[0] != original_scores.numel():
        raise ValueError('timestamps must be flat and align with the score vectors.')
    # Distance to the nearest raw-grid boundary, normalized to [0, 1].
    remainder = np.mod(timestamps.astype(np.float64, copy=False), temporal_bin_size)
    centre_distance = np.minimum(remainder, temporal_bin_size - remainder)
    original_weight = 0.5 + centre_distance / float(temporal_bin_size)
    original_weight = torch.as_tensor(
        original_weight,
        dtype=original_scores.dtype,
        device=original_scores.device,
    )
    return original_scores * original_weight + phase_scores * (1.0 - original_weight)


def blend_temporal_memory_scores(primary_scores, secondary_scores, primary_weight):
    """Blend two aligned temporal-memory score vectors before postprocessing."""
    if primary_scores.shape != secondary_scores.shape:
        raise ValueError(
            'Temporal-memory ensemble score shapes do not match: {} vs {}.'.format(
                tuple(primary_scores.shape), tuple(secondary_scores.shape)
            )
        )
    primary_weight = float(primary_weight)
    if not 0.0 <= primary_weight <= 1.0:
        raise ValueError('primary_weight must be in [0, 1].')
    return (
        primary_scores * primary_weight
        + secondary_scores * (1.0 - primary_weight)
    )
