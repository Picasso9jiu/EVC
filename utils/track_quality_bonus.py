"""Label-free P32 track-quality score bonus.

P32 never creates a new candidate.  It only adds a fixed score bonus to
already-observed events that belong to a seed-supported, temporally smooth
track.  The rule uses only the current video's event coordinates and model
scores, so it is valid when test-video labels are unavailable.
"""

from dataclasses import dataclass

import numpy as np

from utils.postprocess import _as_bool, _spatial_components_in_bin


@dataclass(frozen=True)
class P32TrackQualityBonusConfig:
    """Configuration for the conservative track-quality score bonus."""

    enabled: bool = False
    candidate_floor: float = 0.60
    spatial_radius: int = 2
    temporal_bin_size: int = 50
    max_link_distance: float = 8.0
    max_gap_bins: int = 2
    min_track_bins: int = 4
    min_seed_components: int = 2
    bonus: float = 0.010
    max_score_cap: float = 0.97
    max_motion_residual: float = 2.0
    velocity_history_bins: int = 2

    def __post_init__(self):
        if not 0.0 <= self.candidate_floor <= 1.0:
            raise ValueError('p32_candidate_floor must be in [0, 1].')
        if self.spatial_radius < 0:
            raise ValueError('p32_spatial_radius must be non-negative.')
        if self.temporal_bin_size <= 0:
            raise ValueError('p32_temporal_bin_size must be positive.')
        if self.max_link_distance < 0.0:
            raise ValueError('p32_max_link_distance must be non-negative.')
        if self.max_gap_bins < 1:
            raise ValueError('p32_max_gap_bins must be at least 1.')
        if self.min_track_bins < 2:
            raise ValueError('p32_min_track_bins must be at least 2.')
        if self.min_seed_components < 1:
            raise ValueError('p32_min_seed_components must be positive.')
        if self.bonus <= 0.0:
            raise ValueError('p32_bonus must be positive.')
        if not 0.0 < self.max_score_cap <= 1.0:
            raise ValueError('p32_max_score_cap must be in (0, 1].')
        if self.max_motion_residual <= 0.0:
            raise ValueError('p32_max_motion_residual must be positive.')
        if self.velocity_history_bins < 2:
            raise ValueError('p32_velocity_history_bins must be at least 2.')

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            enabled=_as_bool(
                getattr(cfg, 'p32_track_quality_bonus_enabled', False)
            ),
            candidate_floor=float(getattr(cfg, 'p32_candidate_floor', 0.60)),
            spatial_radius=int(getattr(cfg, 'p32_spatial_radius', 2)),
            temporal_bin_size=int(
                getattr(
                    cfg,
                    'p32_temporal_bin_size',
                    getattr(cfg, 'pd_detT', 50),
                )
            ),
            max_link_distance=float(getattr(cfg, 'p32_max_link_distance', 8.0)),
            max_gap_bins=int(getattr(cfg, 'p32_max_gap_bins', 2)),
            min_track_bins=int(getattr(cfg, 'p32_min_track_bins', 4)),
            min_seed_components=int(
                getattr(cfg, 'p32_min_seed_components', 2)
            ),
            bonus=float(getattr(cfg, 'p32_bonus', 0.010)),
            max_score_cap=float(getattr(cfg, 'p32_max_score_cap', 0.97)),
            max_motion_residual=float(
                getattr(cfg, 'p32_max_motion_residual', 2.0)
            ),
            velocity_history_bins=int(
                getattr(cfg, 'p32_velocity_history_bins', 2)
            ),
        )


@dataclass
class P32TrackQualityBonusStats:
    """Aggregate P32 diagnostics for one or more independent videos."""

    enabled: bool
    input_positive_events: int = 0
    output_positive_events: int = 0
    track_count: int = 0
    eligible_tracks: int = 0
    boosted_events: int = 0
    newly_positive_events: int = 0

    def merge(self, other):
        if self.enabled != other.enabled:
            raise ValueError('Cannot merge enabled and disabled P32 stats.')
        self.input_positive_events += other.input_positive_events
        self.output_positive_events += other.output_positive_events
        self.track_count += other.track_count
        self.eligible_tracks += other.eligible_tracks
        self.boosted_events += other.boosted_events
        self.newly_positive_events += other.newly_positive_events

    def summary(self):
        if not self.enabled:
            return 'disabled (predictions unchanged)'
        return (
            'enabled, positive events: {} -> {}; tracks: {}; eligible: {}; '
            'boosted events: {}; newly positive: {}'
        ).format(
            self.input_positive_events,
            self.output_positive_events,
            self.track_count,
            self.eligible_tracks,
            self.boosted_events,
            self.newly_positive_events,
        )


def _fit_velocity(bin_values, centroids):
    """Fit a constant-velocity model to temporal-bin centroids."""
    bin_values = np.asarray(bin_values, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    if bin_values.shape[0] < 2 or centroids.shape[0] < 2:
        return None
    design = np.column_stack((np.ones_like(bin_values), bin_values))
    velocity_x, _ = np.linalg.lstsq(
        design,
        centroids[:, 0],
        rcond=None,
    )[:2]
    velocity_y, _ = np.linalg.lstsq(
        design,
        centroids[:, 1],
        rcond=None,
    )[:2]
    return float(velocity_x[1]), float(velocity_y[1])


def _linear_residual(bin_values, centroids):
    """Return mean centroid residual under a fitted constant-velocity path."""
    bin_values = np.asarray(bin_values, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    if bin_values.shape[0] < 2:
        return 0.0
    design = np.column_stack((np.ones_like(bin_values), bin_values))
    x_coeff, _, _, _ = np.linalg.lstsq(
        design,
        centroids[:, 0],
        rcond=None,
    )
    y_coeff, _, _, _ = np.linalg.lstsq(
        design,
        centroids[:, 1],
        rcond=None,
    )
    predicted = np.column_stack((design @ x_coeff, design @ y_coeff))
    return float(np.linalg.norm(centroids - predicted, axis=1).mean())


def _bonus_one_video(predictions, coordinates, threshold, config):
    """Return one video's fixed P32 bonus values and diagnostics."""
    event_count = predictions.shape[0]
    stats = P32TrackQualityBonusStats(
        enabled=True,
        input_positive_events=int((predictions >= threshold).sum()),
    )
    bonus_values = np.zeros(event_count, dtype=np.float64)
    seed_mask = predictions >= threshold
    candidate_mask = predictions >= config.candidate_floor
    if not seed_mask.any() or not candidate_mask.any():
        stats.output_positive_events = stats.input_positive_events
        return bonus_values, stats

    active_indices = np.flatnonzero(candidate_mask)
    active_coordinates = coordinates[active_indices]
    temporal_bins = np.floor_divide(
        active_coordinates[:, 2],
        config.temporal_bin_size,
    )
    tracks = []
    velocity_points_by_track = []

    for temporal_bin in np.unique(temporal_bins):
        bin_active_indices = np.flatnonzero(temporal_bins == temporal_bin)
        components = _spatial_components_in_bin(
            active_coordinates[bin_active_indices],
            active_indices[bin_active_indices],
            config.spatial_radius,
        )
        for component in components:
            component['has_seed'] = bool(
                seed_mask[component['event_indices']].any()
            )

        links = []
        for track_index, track in enumerate(tracks):
            bin_difference = int(temporal_bin - track['last_bin'])
            if bin_difference <= 0:
                continue
            velocity = track.get('velocity')
            for component_index, component in enumerate(components):
                last_distance = float(
                    np.linalg.norm(component['centroid'] - track['centroid'])
                )
                if (
                    bin_difference <= config.max_gap_bins
                    and last_distance <= config.max_link_distance
                ):
                    links.append((last_distance, track_index, component_index))
                if velocity is not None:
                    predicted = (
                        track['centroid']
                        + np.asarray(velocity) * bin_difference
                    )
                    predicted_distance = float(
                        np.linalg.norm(component['centroid'] - predicted)
                    )
                    if predicted_distance <= config.max_link_distance * 1.5:
                        links.append(
                            (predicted_distance, track_index, component_index)
                        )

        assigned_tracks = set()
        assigned_components = set()
        for _, track_index, component_index in sorted(links):
            if track_index in assigned_tracks or component_index in assigned_components:
                continue
            track = tracks[track_index]
            component = components[component_index]
            track['components'].append(component)
            track['frame_count'] += 1
            track['has_seed'] = track['has_seed'] or component['has_seed']
            track['centroid'] = component['centroid']
            track['last_bin'] = int(temporal_bin)
            velocity_points_by_track[track_index].append(
                (int(temporal_bin), component['centroid'])
            )
            if len(velocity_points_by_track[track_index]) >= 2:
                recent_points = velocity_points_by_track[track_index][
                    -int(config.velocity_history_bins):
                ]
                track['velocity'] = _fit_velocity(
                    [point[0] for point in recent_points],
                    [point[1] for point in recent_points],
                )
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
                    'velocity': None,
                }
            )
            velocity_points_by_track.append(
                [(int(temporal_bin), component['centroid'])]
            )

    stats.track_count = len(tracks)
    for track_index, track in enumerate(tracks):
        seed_component_count = int(
            sum(1 for component in track['components'] if component['has_seed'])
        )
        if (
            not track['has_seed']
            or track['frame_count'] < config.min_track_bins
            or seed_component_count < config.min_seed_components
        ):
            continue
        points = velocity_points_by_track[track_index]
        residual = _linear_residual(
            np.asarray([point[0] for point in points]),
            np.asarray([point[1] for point in points]),
        )
        if residual > config.max_motion_residual:
            continue
        stats.eligible_tracks += 1
        for component in track['components']:
            indices = np.asarray(component['event_indices'], dtype=np.int64)
            bonus_values[indices] = config.bonus
            stats.boosted_events += int(indices.size)

    output_positive = int(
        (
            np.minimum(config.max_score_cap, predictions + bonus_values)
            >= threshold
        ).sum()
    )
    stats.newly_positive_events = max(
        0,
        output_positive - stats.input_positive_events,
    )
    stats.output_positive_events = output_positive
    return bonus_values, stats


class P32TrackQualityBonus:
    """Apply a fixed score bonus to smooth seed-supported score tracks."""

    def __init__(self, config, prediction_threshold=0.9):
        if not 0.0 <= prediction_threshold <= 1.0:
            raise ValueError('prediction_threshold must be in [0, 1].')
        self.config = config
        self.prediction_threshold = float(prediction_threshold)

    @classmethod
    def from_cfg(cls, cfg, prediction_threshold=0.9):
        return cls(
            P32TrackQualityBonusConfig.from_cfg(cfg),
            prediction_threshold,
        )

    @property
    def enabled(self):
        return self.config.enabled

    def new_stats(self):
        return P32TrackQualityBonusStats(enabled=self.enabled)

    def describe(self):
        if not self.enabled:
            return 'disabled'
        return (
            'enabled (candidate_floor={}, spatial_radius={}, '
            'temporal_bin_size={}, max_link_distance={}, max_gap_bins={}, '
            'min_track_bins={}, min_seed_components={}, bonus={}, '
            'max_score_cap={}, max_motion_residual={})'
        ).format(
            self.config.candidate_floor,
            self.config.spatial_radius,
            self.config.temporal_bin_size,
            self.config.max_link_distance,
            self.config.max_gap_bins,
            self.config.min_track_bins,
            self.config.min_seed_components,
            self.config.bonus,
            self.config.max_score_cap,
            self.config.max_motion_residual,
        )

    def apply(self, predictions, locations):
        """Apply P32 independently to every video represented in ``locations``."""
        if not self.enabled:
            stats = P32TrackQualityBonusStats(enabled=False)
            positive_count = int(
                (predictions.reshape(-1) >= self.prediction_threshold).sum()
            )
            stats.input_positive_events = positive_count
            stats.output_positive_events = positive_count
            return predictions, stats

        import torch

        flattened = predictions.reshape(-1)
        if flattened.numel() != locations.shape[0]:
            raise ValueError(
                'Prediction and location counts do not match: {} and {}.'.format(
                    flattened.numel(), locations.shape[0]
                )
            )
        prediction_values = flattened.detach().cpu().numpy().astype(np.float64)
        location_values = locations.detach().cpu().numpy()
        stats = P32TrackQualityBonusStats(enabled=True)
        bonus_values = np.zeros(prediction_values.shape[0], dtype=np.float64)
        batch_ids = location_values[:, 0].astype(np.int64, copy=False)
        for batch_id in np.unique(batch_ids):
            video_mask = batch_ids == batch_id
            video_bonus, video_stats = _bonus_one_video(
                prediction_values[video_mask],
                location_values[video_mask, 1:4].astype(np.int64, copy=False),
                self.prediction_threshold,
                self.config,
            )
            bonus_values[video_mask] = video_bonus
            stats.merge(video_stats)
        if not bonus_values.any():
            return predictions, stats

        boosted = flattened.clone()
        boosted_values = np.minimum(
            self.config.max_score_cap,
            prediction_values + bonus_values,
        )
        boosted.copy_(torch.from_numpy(boosted_values).to(device=boosted.device))
        return boosted.reshape_as(predictions), stats
