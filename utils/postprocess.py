"""Optional inference-time post-processing modules.

P0 removes small positive-event clusters after the configured decision
threshold. P0b is a motion-aware alternative: it links spatial components in
adjacent temporal bins by centroid distance before filtering short tracks.
Neither module changes scores during training and both are disabled by default.
"""

from dataclasses import dataclass, replace

import numpy as np


def _as_bool(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'true', '1', 'yes', 'on'}:
            return True
        if normalized in {'false', '0', 'no', 'off'}:
            return False
        raise ValueError('Expected a boolean value, got {!r}.'.format(value))
    return bool(value)


@dataclass(frozen=True)
class P0ClusterFilterConfig:
    """Configuration for the optional P0 spatiotemporal cluster filter."""

    enabled: bool = False
    spatial_radius: int = 1
    temporal_bin_size: int = 50
    temporal_radius_bins: int = 1
    min_cluster_events: int = 2
    min_duration_bins: int = 1
    high_confidence_recovery_enabled: bool = False
    retain_min_score: float = 0.98

    def __post_init__(self):
        if self.spatial_radius < 0:
            raise ValueError('p0_spatial_radius must be >= 0.')
        if self.temporal_bin_size <= 0:
            raise ValueError('p0_temporal_bin_size must be > 0.')
        if self.temporal_radius_bins < 0:
            raise ValueError('p0_temporal_radius_bins must be >= 0.')
        if self.min_cluster_events < 1:
            raise ValueError('p0_min_cluster_events must be >= 1.')
        if self.min_duration_bins < 1:
            raise ValueError('p0_min_duration_bins must be >= 1.')
        if not 0.0 <= self.retain_min_score <= 1.0:
            raise ValueError('p0c_retain_min_score must be in [0, 1].')

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(getattr(cfg, 'p0_enabled', False)),
            spatial_radius=int(getattr(cfg, 'p0_spatial_radius', 1)),
            temporal_bin_size=int(
                getattr(cfg, 'p0_temporal_bin_size', getattr(cfg, 'pd_detT', 50))
            ),
            temporal_radius_bins=int(getattr(cfg, 'p0_temporal_radius_bins', 1)),
            min_cluster_events=int(getattr(cfg, 'p0_min_cluster_events', 2)),
            min_duration_bins=int(getattr(cfg, 'p0_min_duration_bins', 1)),
            high_confidence_recovery_enabled=_as_bool(
                getattr(cfg, 'p0c_high_confidence_recovery_enabled', False)
            ),
            retain_min_score=float(getattr(cfg, 'p0c_retain_min_score', 0.98)),
        )


@dataclass(frozen=True)
class P0bTrackFilterConfig:
    """Configuration for the optional P0b centroid-track filter.

    ``max_gap_bins`` is the largest temporal-bin difference between linked
    components. The default value of one therefore links only adjacent bins.
    """

    enabled: bool = False
    spatial_radius: int = 1
    temporal_bin_size: int = 50
    max_link_distance: float = 5.0
    max_gap_bins: int = 1
    min_track_events: int = 3
    min_track_frames: int = 1

    def __post_init__(self):
        if self.spatial_radius < 0:
            raise ValueError('p0b_spatial_radius must be >= 0.')
        if self.temporal_bin_size <= 0:
            raise ValueError('p0b_temporal_bin_size must be > 0.')
        if self.max_link_distance < 0:
            raise ValueError('p0b_max_link_distance must be >= 0.')
        if self.max_gap_bins < 1:
            raise ValueError('p0b_max_gap_bins must be >= 1.')
        if self.min_track_events < 1:
            raise ValueError('p0b_min_track_events must be >= 1.')
        if self.min_track_frames < 1:
            raise ValueError('p0b_min_track_frames must be >= 1.')

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(getattr(cfg, 'p0b_enabled', False)),
            spatial_radius=int(getattr(cfg, 'p0b_spatial_radius', 1)),
            temporal_bin_size=int(
                getattr(cfg, 'p0b_temporal_bin_size', getattr(cfg, 'pd_detT', 50))
            ),
            max_link_distance=float(getattr(cfg, 'p0b_max_link_distance', 5.0)),
            max_gap_bins=int(getattr(cfg, 'p0b_max_gap_bins', 1)),
            min_track_events=int(getattr(cfg, 'p0b_min_track_events', 3)),
            min_track_frames=int(getattr(cfg, 'p0b_min_track_frames', 1)),
        )


@dataclass(frozen=True)
class P18ScoreTrackRecoveryConfig:
    """Recover one weak event from a seed-supported score track.

    The module can be restricted to an observable event-count interval. A
    retained P0/P0c positive is a seed; only lower-score components that form
    a short, spatially continuous track with such a seed can contribute a
    restored event. It never uses labels or video names.
    """

    enabled: bool = False
    event_count_cutoff: int = 100000
    max_event_count: int = 0
    candidate_floor: float = 0.80
    spatial_radius: int = 2
    temporal_bin_size: int = 50
    max_link_distance: float = 6.0
    max_gap_bins: int = 1
    min_track_bins: int = 2
    restore_mode: str = 'best'
    max_restore_events_per_component: int = 0
    velocity_gate_enabled: bool = False
    velocity_gate_base_link_distance: float = 6.0
    velocity_gate_max_acceleration: float = 3.0

    def __post_init__(self):
        if self.event_count_cutoff <= 0:
            raise ValueError('p18_event_count_cutoff must be positive.')
        if self.max_event_count < 0:
            raise ValueError('p18_max_event_count must be non-negative.')
        if not 0.0 <= self.candidate_floor <= 1.0:
            raise ValueError('p18_candidate_floor must be in [0, 1].')
        if self.spatial_radius < 0:
            raise ValueError('p18_spatial_radius must be >= 0.')
        if self.temporal_bin_size <= 0:
            raise ValueError('p18_temporal_bin_size must be > 0.')
        if self.max_link_distance < 0:
            raise ValueError('p18_max_link_distance must be >= 0.')
        if self.max_gap_bins < 1:
            raise ValueError('p18_max_gap_bins must be >= 1.')
        if self.min_track_bins < 2:
            raise ValueError('p18_min_track_bins must be at least 2.')
        if self.restore_mode not in {'best', 'component', 'topk'}:
            raise ValueError(
                "p18_restore_mode must be one of: 'best', 'component', 'topk'."
            )
        if self.max_restore_events_per_component < 0:
            raise ValueError(
                'p18_max_restore_events_per_component must be non-negative.'
            )
        if self.velocity_gate_base_link_distance < 0:
            raise ValueError(
                'p18_velocity_gate_base_link_distance must be >= 0.'
            )
        if self.velocity_gate_base_link_distance > self.max_link_distance:
            raise ValueError(
                'p18_velocity_gate_base_link_distance cannot exceed '
                'p18_max_link_distance.'
            )
        if self.velocity_gate_max_acceleration < 0:
            raise ValueError(
                'p18_velocity_gate_max_acceleration must be >= 0.'
            )
        if self.restore_mode == 'topk' and self.max_restore_events_per_component < 1:
            raise ValueError(
                'p18_max_restore_events_per_component must be positive when '
                'p18_restore_mode=topk.'
            )

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(getattr(cfg, 'p18_score_track_recovery_enabled', False)),
            event_count_cutoff=int(getattr(cfg, 'p18_event_count_cutoff', 100000)),
            max_event_count=int(getattr(cfg, 'p18_max_event_count', 0)),
            candidate_floor=float(getattr(cfg, 'p18_candidate_floor', 0.80)),
            spatial_radius=int(getattr(cfg, 'p18_spatial_radius', 2)),
            temporal_bin_size=int(
                getattr(cfg, 'p18_temporal_bin_size', getattr(cfg, 'pd_detT', 50))
            ),
            max_link_distance=float(getattr(cfg, 'p18_max_link_distance', 6.0)),
            max_gap_bins=int(getattr(cfg, 'p18_max_gap_bins', 1)),
            min_track_bins=int(getattr(cfg, 'p18_min_track_bins', 2)),
            restore_mode=str(getattr(cfg, 'p18_restore_mode', 'best')),
            max_restore_events_per_component=int(
                getattr(cfg, 'p18_max_restore_events_per_component', 0)
            ),
            velocity_gate_enabled=_as_bool(
                getattr(cfg, 'p18_velocity_gate_enabled', False)
            ),
            velocity_gate_base_link_distance=float(
                getattr(cfg, 'p18_velocity_gate_base_link_distance', 6.0)
            ),
            velocity_gate_max_acceleration=float(
                getattr(cfg, 'p18_velocity_gate_max_acceleration', 3.0)
            ),
        )


@dataclass
class P0ClusterFilterStats:
    """Aggregate statistics for one or more filtered batches."""

    enabled: bool
    input_positive_events: int = 0
    output_positive_events: int = 0
    component_count: int = 0
    kept_components: int = 0
    removed_components: int = 0
    recovered_components: int = 0
    recovered_positive_events: int = 0

    @property
    def removed_positive_events(self):
        return self.input_positive_events - self.output_positive_events

    def merge(self, other):
        if self.enabled != other.enabled:
            raise ValueError('Cannot merge enabled and disabled P0 statistics.')
        self.input_positive_events += other.input_positive_events
        self.output_positive_events += other.output_positive_events
        self.component_count += other.component_count
        self.kept_components += other.kept_components
        self.removed_components += other.removed_components
        self.recovered_components += other.recovered_components
        self.recovered_positive_events += other.recovered_positive_events

    def summary(self):
        if not self.enabled:
            return 'disabled (predictions unchanged)'
        summary = (
            'enabled, positive events: {} -> {} (removed {}); '
            'components: {} kept / {} removed'
        ).format(
            self.input_positive_events,
            self.output_positive_events,
            self.removed_positive_events,
            self.kept_components,
            self.removed_components,
        )
        if self.recovered_components:
            summary += '; high-confidence recovery: {} components / {} events'.format(
                self.recovered_components,
                self.recovered_positive_events,
            )
        return summary


@dataclass
class P0bTrackFilterStats:
    """Aggregate statistics for one or more P0b-filtered batches."""

    enabled: bool
    input_positive_events: int = 0
    output_positive_events: int = 0
    component_count: int = 0
    track_count: int = 0
    kept_tracks: int = 0
    removed_tracks: int = 0

    @property
    def removed_positive_events(self):
        return self.input_positive_events - self.output_positive_events

    def merge(self, other):
        if self.enabled != other.enabled:
            raise ValueError('Cannot merge enabled and disabled P0b statistics.')
        self.input_positive_events += other.input_positive_events
        self.output_positive_events += other.output_positive_events
        self.component_count += other.component_count
        self.track_count += other.track_count
        self.kept_tracks += other.kept_tracks
        self.removed_tracks += other.removed_tracks

    def summary(self):
        if not self.enabled:
            return 'disabled (predictions unchanged)'
        return (
            'enabled, positive events: {} -> {} (removed {}); '
            'components: {}; tracks: {} kept / {} removed'
        ).format(
            self.input_positive_events,
            self.output_positive_events,
            self.removed_positive_events,
            self.component_count,
            self.kept_tracks,
            self.removed_tracks,
        )


@dataclass
class P18ScoreTrackRecoveryStats:
    """Aggregate statistics for score-level temporal track recovery."""

    enabled: bool
    eligible_videos: int = 0
    input_positive_events: int = 0
    output_positive_events: int = 0
    candidate_components: int = 0
    track_count: int = 0
    supported_tracks: int = 0
    restored_components: int = 0
    restored_events: int = 0

    def merge(self, other):
        if self.enabled != other.enabled:
            raise ValueError('Cannot merge enabled and disabled P18 statistics.')
        self.eligible_videos += other.eligible_videos
        self.input_positive_events += other.input_positive_events
        self.output_positive_events += other.output_positive_events
        self.candidate_components += other.candidate_components
        self.track_count += other.track_count
        self.supported_tracks += other.supported_tracks
        self.restored_components += other.restored_components
        self.restored_events += other.restored_events

    def summary(self):
        if not self.enabled:
            return 'disabled (predictions unchanged)'
        return (
            'enabled, eligible videos: {}; positive events: {} -> {}; '
            'candidate components: {}; seed-supported tracks: {} / {}; '
            'restored: {} components / {} events'
        ).format(
            self.eligible_videos,
            self.input_positive_events,
            self.output_positive_events,
            self.candidate_components,
            self.supported_tracks,
            self.track_count,
            self.restored_components,
            self.restored_events,
        )


def _filter_one_video(positive_coordinates, config, positive_scores=None):
    """Return an event-level keep mask for one video's positive events."""
    event_count = positive_coordinates.shape[0]
    stats = P0ClusterFilterStats(
        enabled=True,
        input_positive_events=event_count,
    )
    if event_count == 0:
        return np.zeros(0, dtype=bool), stats

    temporal_bins = np.floor_divide(
        positive_coordinates[:, 2], config.temporal_bin_size
    )
    cells = np.column_stack(
        (
            positive_coordinates[:, 0],
            positive_coordinates[:, 1],
            temporal_bins,
        )
    )
    unique_cells, inverse, cell_event_counts = np.unique(
        cells,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    cell_score_max = None
    if config.high_confidence_recovery_enabled:
        if positive_scores is None:
            raise ValueError('P0c high-confidence recovery requires prediction scores.')
        positive_scores = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
        if positive_scores.shape[0] != event_count:
            raise ValueError('positive_scores and positive_coordinates must match.')
        cell_score_max = np.full(len(unique_cells), -np.inf, dtype=np.float64)
        np.maximum.at(cell_score_max, inverse, positive_scores)

    cell_lookup = {
        (int(cell[0]), int(cell[1]), int(cell[2])): index
        for index, cell in enumerate(unique_cells)
    }
    visited = np.zeros(len(unique_cells), dtype=bool)
    keep_cell = np.zeros(len(unique_cells), dtype=bool)
    neighbor_offsets = tuple(
        (dx, dy, dt)
        for dx in range(-config.spatial_radius, config.spatial_radius + 1)
        for dy in range(-config.spatial_radius, config.spatial_radius + 1)
        for dt in range(
            -config.temporal_radius_bins,
            config.temporal_radius_bins + 1,
        )
        if (dx, dy, dt) != (0, 0, 0)
    )

    for start_index in range(len(unique_cells)):
        if visited[start_index]:
            continue

        component = []
        stack = [start_index]
        visited[start_index] = True

        while stack:
            cell_index = stack.pop()
            component.append(cell_index)
            x, y, temporal_bin = unique_cells[cell_index]

            for dx, dy, dt in neighbor_offsets:
                neighbor_index = cell_lookup.get(
                    (int(x + dx), int(y + dy), int(temporal_bin + dt))
                )
                if neighbor_index is not None and not visited[neighbor_index]:
                    visited[neighbor_index] = True
                    stack.append(neighbor_index)

        component = np.asarray(component, dtype=np.int64)
        component_events = int(cell_event_counts[component].sum())
        component_temporal_bins = unique_cells[component, 2]
        component_duration = int(
            component_temporal_bins.max() - component_temporal_bins.min() + 1
        )
        keep_component = (
            component_events >= config.min_cluster_events
            and component_duration >= config.min_duration_bins
        )
        recovered_component = False
        if (
            not keep_component
            and config.high_confidence_recovery_enabled
            and cell_score_max[component].max() >= config.retain_min_score
        ):
            keep_component = True
            recovered_component = True

        stats.component_count += 1
        if keep_component:
            keep_cell[component] = True
            stats.kept_components += 1
            if recovered_component:
                stats.recovered_components += 1
                stats.recovered_positive_events += component_events
        else:
            stats.removed_components += 1

    event_keep_mask = keep_cell[inverse]
    stats.output_positive_events = int(event_keep_mask.sum())
    return event_keep_mask, stats


def filter_positive_events(positive_mask, locations, config, prediction_scores=None):
    """Filter a binary positive mask using independent clusters per video.

    ``locations`` must use the data loader order ``[batch, x, y, t]``. The
    returned mask has the same length as ``positive_mask`` and contains only
    positive events that belong to a retained cluster.
    """
    positive_mask = np.asarray(positive_mask, dtype=bool).reshape(-1)
    locations = np.asarray(locations)

    if locations.ndim != 2 or locations.shape[1] < 4:
        raise ValueError(
            'locations must have shape [N, 4+] ordered as [batch, x, y, t].'
        )
    if locations.shape[0] != positive_mask.shape[0]:
        raise ValueError('positive_mask and locations must have the same length.')

    if prediction_scores is not None:
        prediction_scores = np.asarray(
            prediction_scores,
            dtype=np.float64,
        ).reshape(-1)
        if prediction_scores.shape[0] != positive_mask.shape[0]:
            raise ValueError('prediction_scores and positive_mask must have the same length.')

    if not config.enabled:
        positive_count = int(positive_mask.sum())
        return positive_mask.copy(), P0ClusterFilterStats(
            enabled=False,
            input_positive_events=positive_count,
            output_positive_events=positive_count,
        )

    positive_indices = np.flatnonzero(positive_mask)
    kept_mask = np.zeros_like(positive_mask, dtype=bool)
    stats = P0ClusterFilterStats(enabled=True)
    if positive_indices.size == 0:
        return kept_mask, stats

    positive_locations = locations[positive_indices, :4].astype(np.int64, copy=False)
    positive_scores = (
        prediction_scores[positive_indices]
        if prediction_scores is not None else None
    )
    positive_batch_ids = positive_locations[:, 0]

    for batch_id in np.unique(positive_batch_ids):
        video_event_mask = positive_batch_ids == batch_id
        video_event_indices = positive_indices[video_event_mask]
        video_coordinates = positive_locations[video_event_mask, 1:4]
        video_scores = (
            positive_scores[video_event_mask]
            if positive_scores is not None else None
        )
        video_keep_mask, video_stats = _filter_one_video(
            video_coordinates,
            config,
            video_scores,
        )
        kept_mask[video_event_indices] = video_keep_mask
        stats.merge(video_stats)

    return kept_mask, stats


class P0ClusterFilter:
    """Apply the optional P0 filter to prediction scores without reordering events."""

    def __init__(self, config, prediction_threshold=0.9):
        if not 0.0 <= prediction_threshold <= 1.0:
            raise ValueError('prediction_threshold must be in [0, 1].')
        self.config = config
        self.prediction_threshold = float(prediction_threshold)

    @classmethod
    def from_cfg(cls, cfg, prediction_threshold=0.9):
        return cls(P0ClusterFilterConfig.from_cfg(cfg), prediction_threshold)

    @property
    def enabled(self):
        return self.config.enabled

    def new_stats(self):
        return P0ClusterFilterStats(enabled=self.enabled)

    def describe(self):
        if not self.enabled:
            return 'disabled'
        description = (
            'enabled (spatial_radius={}, temporal_bin_size={}, '
            'temporal_radius_bins={}, min_cluster_events={}, '
            'min_duration_bins={})'
        ).format(
            self.config.spatial_radius,
            self.config.temporal_bin_size,
            self.config.temporal_radius_bins,
            self.config.min_cluster_events,
            self.config.min_duration_bins,
        )
        if self.config.high_confidence_recovery_enabled:
            description += ', P0c high-confidence recovery (retain_min_score={})'.format(
                self.config.retain_min_score
            )
        return description

    def apply(self, predictions, locations):
        """Suppress removed positive scores while retaining all other scores."""
        if not self.enabled:
            return predictions, P0ClusterFilterStats(enabled=False)

        import torch

        flattened_predictions = predictions.reshape(-1)
        if flattened_predictions.numel() != locations.shape[0]:
            raise ValueError(
                'Prediction and location counts do not match: {} and {}.'.format(
                    flattened_predictions.numel(), locations.shape[0]
                )
            )

        prediction_values = flattened_predictions.detach().cpu().numpy()
        location_values = locations.detach().cpu().numpy()
        positive_mask = prediction_values >= self.prediction_threshold
        kept_positive_mask, stats = filter_positive_events(
            positive_mask,
            location_values,
            self.config,
            prediction_scores=prediction_values,
        )

        removed_mask = positive_mask & ~kept_positive_mask
        if not removed_mask.any():
            return predictions, stats

        filtered_predictions = flattened_predictions.clone()
        removed_tensor_mask = torch.from_numpy(removed_mask).to(
            device=filtered_predictions.device
        )
        filtered_predictions[removed_tensor_mask] = 0.0
        return filtered_predictions.reshape_as(predictions), stats


def _spatial_components_in_bin(coordinates, event_indices, spatial_radius):
    """Build spatial components for one temporal bin.

    ``coordinates`` contains only positive events from one video and one time
    bin. Components are formed over unique image cells, while their centroids
    and event counts retain event-level multiplicity.
    """
    unique_cells, inverse = np.unique(
        coordinates[:, :2],
        axis=0,
        return_inverse=True,
    )
    cell_lookup = {
        (int(cell[0]), int(cell[1])): index
        for index, cell in enumerate(unique_cells)
    }
    cell_events = [[] for _ in range(len(unique_cells))]
    for event_index, cell_index in enumerate(inverse):
        cell_events[int(cell_index)].append(event_index)

    neighbor_offsets = tuple(
        (dx, dy)
        for dx in range(-spatial_radius, spatial_radius + 1)
        for dy in range(-spatial_radius, spatial_radius + 1)
        if (dx, dy) != (0, 0)
    )
    visited = np.zeros(len(unique_cells), dtype=bool)
    components = []

    for start_index in range(len(unique_cells)):
        if visited[start_index]:
            continue

        stack = [start_index]
        visited[start_index] = True
        component_cells = []
        component_event_indices = []

        while stack:
            cell_index = stack.pop()
            component_cells.append(cell_index)
            component_event_indices.extend(cell_events[cell_index])
            x, y = unique_cells[cell_index]

            for dx, dy in neighbor_offsets:
                neighbor_index = cell_lookup.get((int(x + dx), int(y + dy)))
                if neighbor_index is not None and not visited[neighbor_index]:
                    visited[neighbor_index] = True
                    stack.append(neighbor_index)

        component_event_indices = np.asarray(component_event_indices, dtype=np.int64)
        component_coordinates = coordinates[component_event_indices, :2]
        components.append(
            {
                'event_indices': event_indices[component_event_indices],
                'centroid': component_coordinates.mean(axis=0),
                'event_count': int(component_event_indices.size),
            }
        )

    return components


def _filter_one_video_by_tracks(positive_coordinates, config):
    """Return an event-level keep mask using centroid-linked tracks."""
    event_count = positive_coordinates.shape[0]
    stats = P0bTrackFilterStats(
        enabled=True,
        input_positive_events=event_count,
    )
    if event_count == 0:
        return np.zeros(0, dtype=bool), stats

    temporal_bins = np.floor_divide(
        positive_coordinates[:, 2], config.temporal_bin_size
    )
    tracks = []

    for temporal_bin in np.unique(temporal_bins):
        bin_event_indices = np.flatnonzero(temporal_bins == temporal_bin)
        components = _spatial_components_in_bin(
            positive_coordinates[bin_event_indices],
            bin_event_indices,
            config.spatial_radius,
        )
        stats.component_count += len(components)

        candidate_links = []
        for track_index, track in enumerate(tracks):
            bin_difference = int(temporal_bin - track['last_bin'])
            if not 1 <= bin_difference <= config.max_gap_bins:
                continue
            for component_index, component in enumerate(components):
                distance = float(
                    np.linalg.norm(component['centroid'] - track['centroid'])
                )
                if distance <= config.max_link_distance:
                    candidate_links.append((distance, track_index, component_index))

        assigned_tracks = set()
        assigned_components = set()
        for _, track_index, component_index in sorted(candidate_links):
            if track_index in assigned_tracks or component_index in assigned_components:
                continue

            track = tracks[track_index]
            component = components[component_index]
            track['event_indices'].append(component['event_indices'])
            track['event_count'] += component['event_count']
            track['frame_count'] += 1
            track['centroid'] = component['centroid']
            track['last_bin'] = int(temporal_bin)
            assigned_tracks.add(track_index)
            assigned_components.add(component_index)

        for component_index, component in enumerate(components):
            if component_index in assigned_components:
                continue
            tracks.append(
                {
                    'event_indices': [component['event_indices']],
                    'event_count': component['event_count'],
                    'frame_count': 1,
                    'centroid': component['centroid'],
                    'last_bin': int(temporal_bin),
                }
            )

    event_keep_mask = np.zeros(event_count, dtype=bool)
    stats.track_count = len(tracks)
    for track in tracks:
        keep_track = (
            track['event_count'] >= config.min_track_events
            and track['frame_count'] >= config.min_track_frames
        )
        if keep_track:
            event_keep_mask[np.concatenate(track['event_indices'])] = True
            stats.kept_tracks += 1
        else:
            stats.removed_tracks += 1

    stats.output_positive_events = int(event_keep_mask.sum())
    return event_keep_mask, stats


def filter_positive_events_by_tracks(positive_mask, locations, config):
    """Filter a positive mask independently for each video with P0b tracks."""
    positive_mask = np.asarray(positive_mask, dtype=bool).reshape(-1)
    locations = np.asarray(locations)

    if locations.ndim != 2 or locations.shape[1] < 4:
        raise ValueError(
            'locations must have shape [N, 4+] ordered as [batch, x, y, t].'
        )
    if locations.shape[0] != positive_mask.shape[0]:
        raise ValueError('positive_mask and locations must have the same length.')

    if not config.enabled:
        positive_count = int(positive_mask.sum())
        return positive_mask.copy(), P0bTrackFilterStats(
            enabled=False,
            input_positive_events=positive_count,
            output_positive_events=positive_count,
        )

    positive_indices = np.flatnonzero(positive_mask)
    kept_mask = np.zeros_like(positive_mask, dtype=bool)
    stats = P0bTrackFilterStats(enabled=True)
    if positive_indices.size == 0:
        return kept_mask, stats

    positive_locations = locations[positive_indices, :4].astype(np.int64, copy=False)
    positive_batch_ids = positive_locations[:, 0]
    for batch_id in np.unique(positive_batch_ids):
        video_event_mask = positive_batch_ids == batch_id
        video_event_indices = positive_indices[video_event_mask]
        video_coordinates = positive_locations[video_event_mask, 1:4]
        video_keep_mask, video_stats = _filter_one_video_by_tracks(
            video_coordinates,
            config,
        )
        kept_mask[video_event_indices] = video_keep_mask
        stats.merge(video_stats)

    return kept_mask, stats


class P0bTrackFilter:
    """Apply optional P0b centroid-track filtering to prediction scores."""

    def __init__(self, config, prediction_threshold=0.9):
        if not 0.0 <= prediction_threshold <= 1.0:
            raise ValueError('prediction_threshold must be in [0, 1].')
        self.config = config
        self.prediction_threshold = float(prediction_threshold)

    @classmethod
    def from_cfg(cls, cfg, prediction_threshold=0.9):
        return cls(P0bTrackFilterConfig.from_cfg(cfg), prediction_threshold)

    @property
    def enabled(self):
        return self.config.enabled

    def new_stats(self):
        return P0bTrackFilterStats(enabled=self.enabled)

    def describe(self):
        if not self.enabled:
            return 'disabled'
        return (
            'enabled (spatial_radius={}, temporal_bin_size={}, '
            'max_link_distance={}, max_gap_bins={}, min_track_events={}, '
            'min_track_frames={})'
        ).format(
            self.config.spatial_radius,
            self.config.temporal_bin_size,
            self.config.max_link_distance,
            self.config.max_gap_bins,
            self.config.min_track_events,
            self.config.min_track_frames,
        )

    def apply(self, predictions, locations):
        """Suppress positive scores that do not belong to a retained track."""
        if not self.enabled:
            return predictions, P0bTrackFilterStats(enabled=False)

        import torch

        flattened_predictions = predictions.reshape(-1)
        if flattened_predictions.numel() != locations.shape[0]:
            raise ValueError(
                'Prediction and location counts do not match: {} and {}.'.format(
                    flattened_predictions.numel(), locations.shape[0]
                )
            )

        prediction_values = flattened_predictions.detach().cpu().numpy()
        location_values = locations.detach().cpu().numpy()
        positive_mask = prediction_values >= self.prediction_threshold
        kept_positive_mask, stats = filter_positive_events_by_tracks(
            positive_mask,
            location_values,
            self.config,
        )

        removed_mask = positive_mask & ~kept_positive_mask
        if not removed_mask.any():
            return predictions, stats

        filtered_predictions = flattened_predictions.clone()
        removed_tensor_mask = torch.from_numpy(removed_mask).to(
            device=filtered_predictions.device
        )
        filtered_predictions[removed_tensor_mask] = 0.0
        return filtered_predictions.reshape_as(predictions), stats


def _track_passes_velocity_gate(track, config):
    """Keep normal P18 tracks; vet only links beyond the base distance."""
    if not config.velocity_gate_enabled:
        return True

    components = track['components']
    centers = np.asarray(
        [component['centroid'] for component in components],
        dtype=np.float64,
    )
    bins = np.asarray(
        [component['temporal_bin'] for component in components],
        dtype=np.float64,
    )
    time_deltas = np.diff(bins)
    velocities = np.diff(centers, axis=0) / time_deltas[:, None]
    speeds = np.linalg.norm(velocities, axis=1)
    if not (speeds > config.velocity_gate_base_link_distance).any():
        return True
    if velocities.shape[0] < 2:
        return False
    accelerations = np.linalg.norm(np.diff(velocities, axis=0), axis=1)
    return bool(
        np.all(accelerations <= config.velocity_gate_max_acceleration)
    )


def _recover_one_video_by_score_tracks_core(
    prediction_scores,
    coordinates,
    config,
    prediction_threshold,
):
    """Find weak events connected to a retained score-track seed.

    ``prediction_scores`` are the scores after P0/P0c.  This matters: P0
    removals are zeroed and therefore cannot create a recovery seed.
    """
    event_count = prediction_scores.shape[0]
    input_positive_count = int((prediction_scores >= prediction_threshold).sum())
    stats = P18ScoreTrackRecoveryStats(
        enabled=True,
        input_positive_events=input_positive_count,
        output_positive_events=input_positive_count,
    )
    recovery_mask = np.zeros(event_count, dtype=bool)
    if (
        event_count <= config.event_count_cutoff
        or (config.max_event_count and event_count > config.max_event_count)
    ):
        return recovery_mask, stats

    stats.eligible_videos = 1
    seed_mask = prediction_scores >= prediction_threshold
    weak_mask = (
        (prediction_scores >= config.candidate_floor)
        & ~seed_mask
    )
    candidate_mask = seed_mask | weak_mask
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0 or not seed_mask.any() or not weak_mask.any():
        return recovery_mask, stats

    candidate_coordinates = coordinates[candidate_indices]
    temporal_bins = np.floor_divide(
        candidate_coordinates[:, 2],
        config.temporal_bin_size,
    )
    tracks = []

    for temporal_bin in np.unique(temporal_bins):
        bin_candidate_indices = np.flatnonzero(temporal_bins == temporal_bin)
        components = _spatial_components_in_bin(
            candidate_coordinates[bin_candidate_indices],
            candidate_indices[bin_candidate_indices],
            config.spatial_radius,
        )
        for component in components:
            component['temporal_bin'] = int(temporal_bin)
            component_scores = prediction_scores[component['event_indices']]
            score_order = np.argsort(component_scores)[::-1]
            component['sorted_event_indices'] = component['event_indices'][score_order]
            component['has_seed'] = bool(
                seed_mask[component['event_indices']].any()
            )
        stats.candidate_components += len(components)

        candidate_links = []
        for track_index, track in enumerate(tracks):
            bin_difference = int(temporal_bin - track['last_bin'])
            if not 1 <= bin_difference <= config.max_gap_bins:
                continue
            for component_index, component in enumerate(components):
                distance = float(
                    np.linalg.norm(component['centroid'] - track['centroid'])
                )
                if distance <= config.max_link_distance:
                    candidate_links.append((distance, track_index, component_index))

        assigned_tracks = set()
        assigned_components = set()
        for _, track_index, component_index in sorted(candidate_links):
            if track_index in assigned_tracks or component_index in assigned_components:
                continue
            track = tracks[track_index]
            component = components[component_index]
            track['components'].append(component)
            track['frame_count'] += 1
            track['has_seed'] = track['has_seed'] or component['has_seed']
            track['centroid'] = component['centroid']
            track['last_bin'] = int(temporal_bin)
            assigned_tracks.add(track_index)
            assigned_components.add(component_index)

        for component_index, component in enumerate(components):
            if component_index in assigned_components:
                continue
            tracks.append(
                {
                    'components': [component],
                    'frame_count': 1,
                    'has_seed': component['has_seed'],
                    'centroid': component['centroid'],
                    'last_bin': int(temporal_bin),
                }
            )

    stats.track_count = len(tracks)
    for track in tracks:
        if not (
            track['has_seed']
            and track['frame_count'] >= config.min_track_bins
        ):
            continue
        if not _track_passes_velocity_gate(track, config):
            continue
        stats.supported_tracks += 1
        for component in track['components']:
            if component['has_seed']:
                continue
            if config.restore_mode == 'best':
                recovered_indices = component['sorted_event_indices'][:1]
            elif config.restore_mode == 'topk':
                recovered_indices = component['sorted_event_indices'][
                    :config.max_restore_events_per_component
                ]
            else:
                recovered_indices = component['event_indices']
                if config.max_restore_events_per_component:
                    recovered_indices = component['sorted_event_indices'][
                        :config.max_restore_events_per_component
                    ]
            if recovered_indices.size == 0:
                continue
            recovery_mask[recovered_indices] = True
            stats.restored_components += 1

    stats.restored_events = int(recovery_mask.sum())
    stats.output_positive_events += stats.restored_events
    return recovery_mask, stats


def _recover_one_video_by_score_tracks(
    prediction_scores,
    coordinates,
    config,
    prediction_threshold,
):
    """Recover normal P18 tracks plus optional, gated high-speed tracks."""
    if (
        not config.velocity_gate_enabled
        or config.max_link_distance <= config.velocity_gate_base_link_distance
    ):
        return _recover_one_video_by_score_tracks_core(
            prediction_scores,
            coordinates,
            config,
            prediction_threshold,
        )

    # A wider greedy matching radius can merge a normal short track into an
    # unrelated long link. Keep the production-radius result as an immutable
    # base and add only gated wide-radius recoveries on top of it.
    base_config = replace(
        config,
        velocity_gate_enabled=False,
        max_link_distance=config.velocity_gate_base_link_distance,
    )
    base_mask, base_stats = _recover_one_video_by_score_tracks_core(
        prediction_scores,
        coordinates,
        base_config,
        prediction_threshold,
    )
    extended_mask, extended_stats = _recover_one_video_by_score_tracks_core(
        prediction_scores,
        coordinates,
        config,
        prediction_threshold,
    )
    recovery_mask = base_mask | extended_mask
    extended_stats.restored_components = int(recovery_mask.sum())
    extended_stats.restored_events = int(recovery_mask.sum())
    extended_stats.input_positive_events = base_stats.input_positive_events
    extended_stats.output_positive_events = (
        extended_stats.input_positive_events + extended_stats.restored_events
    )
    return recovery_mask, extended_stats


def recover_seed_supported_track_events(
    prediction_scores,
    locations,
    config,
    prediction_threshold,
):
    """Return a mask of label-free P18 weak-event recoveries.

    Locations follow the project convention ``[batch, x, y, t]``.  Each batch
    member is treated as an independent video, so a recovered track never
    crosses a video boundary.
    """
    prediction_scores = np.asarray(prediction_scores, dtype=np.float64).reshape(-1)
    locations = np.asarray(locations)
    if locations.ndim != 2 or locations.shape[1] < 4:
        raise ValueError(
            'locations must have shape [N, 4+] ordered as [batch, x, y, t].'
        )
    if locations.shape[0] != prediction_scores.shape[0]:
        raise ValueError('prediction_scores and locations must have the same length.')
    if not 0.0 <= prediction_threshold <= 1.0:
        raise ValueError('prediction_threshold must be in [0, 1].')

    input_positive_count = int((prediction_scores >= prediction_threshold).sum())
    if not config.enabled:
        return np.zeros(prediction_scores.shape[0], dtype=bool), P18ScoreTrackRecoveryStats(
            enabled=False,
            input_positive_events=input_positive_count,
            output_positive_events=input_positive_count,
        )

    recovery_mask = np.zeros(prediction_scores.shape[0], dtype=bool)
    stats = P18ScoreTrackRecoveryStats(enabled=True)
    batch_ids = locations[:, 0].astype(np.int64, copy=False)
    for batch_id in np.unique(batch_ids):
        video_indices = np.flatnonzero(batch_ids == batch_id)
        video_recovery_mask, video_stats = _recover_one_video_by_score_tracks(
            prediction_scores[video_indices],
            locations[video_indices, 1:4].astype(np.int64, copy=False),
            config,
            prediction_threshold,
        )
        recovery_mask[video_indices] = video_recovery_mask
        stats.merge(video_stats)

    return recovery_mask, stats


class P18ScoreTrackRecovery:
    """Restore one weak event for seed-supported dense-video tracks."""

    def __init__(self, config, prediction_threshold=0.9):
        if not 0.0 <= prediction_threshold <= 1.0:
            raise ValueError('prediction_threshold must be in [0, 1].')
        self.config = config
        self.prediction_threshold = float(prediction_threshold)

    @classmethod
    def from_cfg(cls, cfg, prediction_threshold=0.9):
        return cls(
            P18ScoreTrackRecoveryConfig.from_cfg(cfg),
            prediction_threshold,
        )

    @property
    def enabled(self):
        return self.config.enabled

    def new_stats(self):
        return P18ScoreTrackRecoveryStats(enabled=self.enabled)

    def describe(self):
        if not self.enabled:
            return 'disabled'
        return (
            'enabled (event_count > {} and <= {}, candidate_floor={}, spatial_radius={}, '
            'temporal_bin_size={}, max_link_distance={}, max_gap_bins={}, '
            'min_track_bins={}, restore_mode={}, max_restore_events_per_component={}, '
            'velocity_gate_enabled={}, velocity_gate_base_link_distance={}, '
            'velocity_gate_max_acceleration={})'
        ).format(
            self.config.event_count_cutoff,
            self.config.max_event_count or 'unbounded',
            self.config.candidate_floor,
            self.config.spatial_radius,
            self.config.temporal_bin_size,
            self.config.max_link_distance,
            self.config.max_gap_bins,
            self.config.min_track_bins,
            self.config.restore_mode,
            self.config.max_restore_events_per_component,
            self.config.velocity_gate_enabled,
            self.config.velocity_gate_base_link_distance,
            self.config.velocity_gate_max_acceleration,
        )

    def apply(self, predictions, locations):
        """Raise recovered weak scores to the current decision threshold."""
        if not self.enabled:
            return predictions, P18ScoreTrackRecoveryStats(enabled=False)

        import torch

        flattened_predictions = predictions.reshape(-1)
        if flattened_predictions.numel() != locations.shape[0]:
            raise ValueError(
                'Prediction and location counts do not match: {} and {}.'.format(
                    flattened_predictions.numel(), locations.shape[0]
                )
            )
        recovery_mask, stats = recover_seed_supported_track_events(
            flattened_predictions.detach().cpu().numpy(),
            locations.detach().cpu().numpy(),
            self.config,
            self.prediction_threshold,
        )
        if not recovery_mask.any():
            return predictions, stats

        recovered_predictions = flattened_predictions.clone()
        recovery_tensor_mask = torch.from_numpy(recovery_mask).to(
            device=recovered_predictions.device
        )
        recovered_predictions[recovery_tensor_mask] = self.prediction_threshold
        return recovered_predictions.reshape_as(predictions), stats


@dataclass
class ChallengePostprocessStats:
    """Composite P0/P0b and optional P18 inference statistics."""

    base_stats: object
    recovery_stats: P18ScoreTrackRecoveryStats

    def __getattr__(self, name):
        """Preserve the base-statistics interface for diagnostic callers."""
        return getattr(self.base_stats, name)

    def merge(self, other):
        self.base_stats.merge(other.base_stats)
        self.recovery_stats.merge(other.recovery_stats)

    def summary(self):
        summary = self.base_stats.summary()
        if self.recovery_stats.enabled:
            summary += '; P18 score-track recovery: {}'.format(
                self.recovery_stats.summary()
            )
        return summary


class ChallengePostprocessor:
    """Apply P0/P0b followed by optional P18 score-track recovery.

    P0 and P0b have overlapping purposes. Requiring one at a time keeps their
    validation result attributable to a single, reproducible configuration.
    P18 is intentionally a separate recovery stage: it can use P0-retained
    positives as track seeds but cannot reintroduce P0-suppressed seed noise.
    """

    def __init__(self, postprocessor, score_track_recovery):
        self._postprocessor = postprocessor
        self._score_track_recovery = score_track_recovery

    @classmethod
    def from_cfg(cls, cfg, prediction_threshold=0.9):
        p0_filter = P0ClusterFilter.from_cfg(cfg, prediction_threshold)
        p0b_filter = P0bTrackFilter.from_cfg(cfg, prediction_threshold)
        score_track_recovery = P18ScoreTrackRecovery.from_cfg(
            cfg,
            prediction_threshold,
        )
        if (
            p0_filter.config.high_confidence_recovery_enabled
            and not p0_filter.enabled
        ):
            raise ValueError(
                'P0c high-confidence recovery requires POSTPROCESS.p0_enabled=true.'
            )
        if p0_filter.enabled and p0b_filter.enabled:
            raise ValueError(
                'P0 and P0b cannot be enabled together. Choose one postprocessor.'
            )
        if score_track_recovery.enabled and not p0_filter.enabled:
            raise ValueError(
                'P18 score-track recovery requires POSTPROCESS.p0_enabled=true.'
            )
        return cls(
            p0b_filter if p0b_filter.enabled else p0_filter,
            score_track_recovery,
        )

    @property
    def enabled(self):
        return self._postprocessor.enabled or self._score_track_recovery.enabled

    def new_stats(self):
        return ChallengePostprocessStats(
            self._postprocessor.new_stats(),
            self._score_track_recovery.new_stats(),
        )

    def describe(self):
        description = self._postprocessor.describe()
        if self._score_track_recovery.enabled:
            description += '; P18 score-track recovery: {}'.format(
                self._score_track_recovery.describe()
            )
        return description

    def apply(self, predictions, locations):
        predictions, base_stats = self._postprocessor.apply(predictions, locations)
        predictions, recovery_stats = self._score_track_recovery.apply(
            predictions,
            locations,
        )
        return predictions, ChallengePostprocessStats(base_stats, recovery_stats)
