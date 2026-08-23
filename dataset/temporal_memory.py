"""Sequence views for the bidirectional full-stream temporal memory model."""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.temporal_frame import (
    build_temporal_context_frame,
    load_temporal_frame_video,
    temporal_phase_shift_temporal_frame_video,
)
from dataset.trajectory_augmentation import augment_target_trajectory
from dataset.cross_video_copy_paste import copy_paste_target_track


def temporal_sequence_start(center_bin, bin_count, sequence_length):
    """Choose a fixed-length sequence centred on an observed time bin."""
    center_bin = int(center_bin)
    bin_count = int(bin_count)
    sequence_length = int(sequence_length)
    if bin_count <= 0 or sequence_length <= 0:
        raise ValueError('bin_count and sequence_length must be positive.')
    if sequence_length > bin_count:
        raise ValueError('sequence_length must not exceed bin_count.')
    if center_bin < 0 or center_bin >= bin_count:
        raise ValueError('center_bin is outside the available range.')
    return min(
        max(center_bin - sequence_length // 2, 0),
        bin_count - sequence_length,
    )


def event_count_matches_training_route(
    event_count,
    dense_only_enabled=False,
    dense_only_event_count_cutoff=100000,
    low_density_only_enabled=False,
    low_density_only_event_count_cutoff=30000,
):
    """Return whether an event count belongs to the selected training route."""
    if dense_only_enabled and low_density_only_enabled:
        raise ValueError('dense-only and low-density-only training are exclusive.')
    if dense_only_enabled:
        return int(event_count) > int(dense_only_event_count_cutoff)
    if low_density_only_enabled:
        return int(event_count) <= int(low_density_only_event_count_cutoff)
    return True


def horizontal_flip_temporal_memory_sequence(frames, event_x, width):
    """Mirror one sequence and its aligned event x coordinates."""
    frames = np.asarray(frames)
    event_x = np.asarray(event_x)
    width = int(width)
    if frames.ndim != 4:
        raise ValueError('frames must have shape [T, C, H, W].')
    if width <= 0 or frames.shape[-1] != width:
        raise ValueError('width must match the frame width and be positive.')
    if event_x.ndim != 1:
        raise ValueError('event_x must be one-dimensional.')
    if event_x.size and (event_x.min() < 0 or event_x.max() >= width):
        raise ValueError('event x coordinates are outside the configured width.')
    return frames[..., ::-1].copy(), width - 1 - event_x


class TemporalMemoryTrainDataset(Dataset):
    """Sample contiguous full-stream frame sequences without validation data."""

    def __init__(
        self,
        root,
        whole_t,
        temporal_bin_size,
        context_bins,
        sequence_length,
        width,
        height,
        views_per_video,
        positive_frame_probability,
        random_seed,
        log_count_clip=4.0,
        cache_all_videos=True,
        cache_video_count=16,
        dense_sampling_enabled=False,
        dense_event_count_cutoff=200000,
        dense_view_multiplier=2,
        mid_density_sampling_enabled=False,
        mid_density_min_event_count=30000,
        mid_density_max_event_count=200000,
        mid_density_view_multiplier=2,
        dense_only_enabled=False,
        dense_only_event_count_cutoff=100000,
        low_density_only_enabled=False,
        low_density_only_event_count_cutoff=30000,
        max_videos_per_epoch=0,
        motion_sampling_enabled=False,
        motion_sampling_min_event_count=30000,
        motion_sampling_min_displacement=4.0,
        motion_sampling_probability=0.50,
        motion_sampling_extra_views_only=True,
        trajectory_augmentation_enabled=False,
        trajectory_augmentation_min_event_count=30000,
        trajectory_augmentation_probability=0.50,
        trajectory_augmentation_extra_views_only=True,
        trajectory_augmentation_residual_speed=4.0,
        trajectory_augmentation_min_track_bins=3,
        cross_video_copy_paste_enabled=False,
        cross_video_copy_paste_min_event_count=30000,
        cross_video_copy_paste_probability=0.25,
        cross_video_copy_paste_extra_views_only=True,
        cross_video_copy_paste_extra_views=0,
        cross_video_copy_paste_min_track_bins=3,
        cross_video_copy_paste_collision_radius=1,
        horizontal_flip_augmentation_enabled=False,
        horizontal_flip_augmentation_probability=0.50,
        temporal_phase_offset=0,
        local_temporal_context_enabled=False,
        local_temporal_context_kernel_size=11,
        fold_manifest_path='',
        train_folds='',
    ):
        self.root = Path(root)
        self.whole_t = int(whole_t)
        self.temporal_bin_size = int(temporal_bin_size)
        self.context_bins = int(context_bins)
        self.sequence_length = int(sequence_length)
        self.width = int(width)
        self.height = int(height)
        self.views_per_video = int(views_per_video)
        self.positive_frame_probability = float(positive_frame_probability)
        self.random_seed = int(random_seed)
        self.log_count_clip = float(log_count_clip)
        self.cache_all_videos = bool(cache_all_videos)
        self.cache_video_count = int(cache_video_count)
        self.dense_sampling_enabled = bool(dense_sampling_enabled)
        self.dense_event_count_cutoff = int(dense_event_count_cutoff)
        self.dense_view_multiplier = int(dense_view_multiplier)
        self.mid_density_sampling_enabled = bool(mid_density_sampling_enabled)
        self.mid_density_min_event_count = int(mid_density_min_event_count)
        self.mid_density_max_event_count = int(mid_density_max_event_count)
        self.mid_density_view_multiplier = int(mid_density_view_multiplier)
        self.dense_only_enabled = bool(dense_only_enabled)
        self.dense_only_event_count_cutoff = int(dense_only_event_count_cutoff)
        self.low_density_only_enabled = bool(low_density_only_enabled)
        self.low_density_only_event_count_cutoff = int(
            low_density_only_event_count_cutoff
        )
        self.max_videos_per_epoch = int(max_videos_per_epoch)
        self.motion_sampling_enabled = bool(motion_sampling_enabled)
        self.motion_sampling_min_event_count = int(motion_sampling_min_event_count)
        self.motion_sampling_min_displacement = float(
            motion_sampling_min_displacement
        )
        self.motion_sampling_probability = float(motion_sampling_probability)
        self.motion_sampling_extra_views_only = bool(
            motion_sampling_extra_views_only
        )
        self.trajectory_augmentation_enabled = bool(
            trajectory_augmentation_enabled
        )
        self.trajectory_augmentation_min_event_count = int(
            trajectory_augmentation_min_event_count
        )
        self.trajectory_augmentation_probability = float(
            trajectory_augmentation_probability
        )
        self.trajectory_augmentation_extra_views_only = bool(
            trajectory_augmentation_extra_views_only
        )
        self.trajectory_augmentation_residual_speed = float(
            trajectory_augmentation_residual_speed
        )
        self.trajectory_augmentation_min_track_bins = int(
            trajectory_augmentation_min_track_bins
        )
        self.cross_video_copy_paste_enabled = bool(
            cross_video_copy_paste_enabled
        )
        self.cross_video_copy_paste_min_event_count = int(
            cross_video_copy_paste_min_event_count
        )
        self.cross_video_copy_paste_probability = float(
            cross_video_copy_paste_probability
        )
        self.cross_video_copy_paste_extra_views_only = bool(
            cross_video_copy_paste_extra_views_only
        )
        self.cross_video_copy_paste_extra_views = int(
            cross_video_copy_paste_extra_views
        )
        self.cross_video_copy_paste_min_track_bins = int(
            cross_video_copy_paste_min_track_bins
        )
        self.cross_video_copy_paste_collision_radius = int(
            cross_video_copy_paste_collision_radius
        )
        self.horizontal_flip_augmentation_enabled = bool(
            horizontal_flip_augmentation_enabled
        )
        self.horizontal_flip_augmentation_probability = float(
            horizontal_flip_augmentation_probability
        )
        self.temporal_phase_offset = int(temporal_phase_offset)
        self.local_temporal_context_enabled = bool(local_temporal_context_enabled)
        self.local_temporal_context_kernel_size = int(
            local_temporal_context_kernel_size
        )
        self.fold_manifest_path = str(fold_manifest_path or '').strip()
        self.train_folds = str(train_folds or '').strip()
        self.current_epoch = 0

        self.file_paths = sorted(self.root.glob('*.npz'))
        if not self.file_paths:
            raise RuntimeError('No npz files found in {}'.format(self.root))
        if self.fold_manifest_path or self.train_folds:
            if not self.fold_manifest_path or not self.train_folds:
                raise ValueError(
                    'fold_manifest_path and train_folds must be provided together.'
                )
            requested_folds = set()
            for token in self.train_folds.split(','):
                token = token.strip()
                if token:
                    requested_folds.add(int(token))
            if not requested_folds:
                raise ValueError('train_folds must contain at least one fold.')
            with Path(self.fold_manifest_path).open('r', encoding='utf-8') as stream:
                manifest = json.load(stream)
            records = manifest.get('records')
            if not isinstance(records, list) or not records:
                raise ValueError('Fold manifest records are missing or empty.')
            fold_by_name = {}
            for record in records:
                name = str(record.get('name', '')).strip()
                if not name or 'fold' not in record:
                    raise ValueError('Fold manifest contains an invalid record.')
                if name in fold_by_name:
                    raise ValueError('Fold manifest contains duplicate video names.')
                fold_by_name[name] = int(record['fold'])
            missing_manifest_names = [
                path.name for path in self.file_paths if path.name not in fold_by_name
            ]
            if missing_manifest_names:
                raise ValueError(
                    'Fold manifest does not cover training videos: {}'.format(
                        missing_manifest_names[:5]
                    )
                )
            self.file_paths = [
                path
                for path in self.file_paths
                if fold_by_name[path.name] in requested_folds
            ]
            if not self.file_paths:
                raise RuntimeError(
                    'No training videos belong to requested folds {}.'.format(
                        sorted(requested_folds)
                    )
                )
            self.selected_folds = tuple(sorted(requested_folds))
        else:
            self.selected_folds = None
        if self.context_bins < 1 or self.context_bins % 2 == 0:
            raise ValueError('context_bins must be a positive odd integer.')
        if self.sequence_length <= 0:
            raise ValueError('sequence_length must be positive.')
        if self.views_per_video <= 0:
            raise ValueError('views_per_video must be positive.')
        if not 0.0 <= self.positive_frame_probability <= 1.0:
            raise ValueError('positive_frame_probability must be in [0, 1].')
        if self.cache_video_count <= 0:
            raise ValueError('cache_video_count must be positive.')
        if self.dense_sampling_enabled:
            if self.dense_event_count_cutoff <= 0:
                raise ValueError('dense_event_count_cutoff must be positive.')
            if self.dense_view_multiplier < 2:
                raise ValueError('dense_view_multiplier must be at least two.')
        if self.mid_density_sampling_enabled:
            if self.mid_density_min_event_count < 0:
                raise ValueError(
                    'mid_density_min_event_count must be non-negative.'
                )
            if self.mid_density_max_event_count <= self.mid_density_min_event_count:
                raise ValueError(
                    'mid_density_max_event_count must exceed the minimum.'
                )
            if self.mid_density_view_multiplier < 2:
                raise ValueError(
                    'mid_density_view_multiplier must be at least two.'
                )
        if self.dense_only_enabled and self.dense_only_event_count_cutoff <= 0:
            raise ValueError('dense_only_event_count_cutoff must be positive.')
        if (
            self.low_density_only_enabled
            and self.low_density_only_event_count_cutoff <= 0
        ):
            raise ValueError('low_density_only_event_count_cutoff must be positive.')
        if self.dense_only_enabled and self.low_density_only_enabled:
            raise ValueError('dense-only and low-density-only training are exclusive.')
        if self.max_videos_per_epoch < 0:
            raise ValueError('max_videos_per_epoch must be non-negative.')
        if self.motion_sampling_min_event_count < 0:
            raise ValueError('motion_sampling_min_event_count must be non-negative.')
        if self.motion_sampling_min_displacement < 0.0:
            raise ValueError('motion_sampling_min_displacement must be non-negative.')
        if not 0.0 <= self.motion_sampling_probability <= 1.0:
            raise ValueError('motion_sampling_probability must be in [0, 1].')
        if self.trajectory_augmentation_min_event_count < 0:
            raise ValueError(
                'trajectory_augmentation_min_event_count must be non-negative.'
            )
        if not 0.0 <= self.trajectory_augmentation_probability <= 1.0:
            raise ValueError(
                'trajectory_augmentation_probability must be in [0, 1].'
            )
        if self.trajectory_augmentation_residual_speed <= 0.0:
            raise ValueError(
                'trajectory_augmentation_residual_speed must be positive.'
            )
        if self.trajectory_augmentation_min_track_bins < 2:
            raise ValueError(
                'trajectory_augmentation_min_track_bins must be at least two.'
            )
        if self.cross_video_copy_paste_min_event_count < 0:
            raise ValueError(
                'cross_video_copy_paste_min_event_count must be non-negative.'
            )
        if not 0.0 <= self.cross_video_copy_paste_probability <= 1.0:
            raise ValueError(
                'cross_video_copy_paste_probability must be in [0, 1].'
            )
        if self.cross_video_copy_paste_extra_views < 0:
            raise ValueError(
                'cross_video_copy_paste_extra_views must be non-negative.'
            )
        if self.cross_video_copy_paste_min_track_bins < 2:
            raise ValueError(
                'cross_video_copy_paste_min_track_bins must be at least two.'
            )
        if self.cross_video_copy_paste_collision_radius < 0:
            raise ValueError(
                'cross_video_copy_paste_collision_radius must be non-negative.'
            )
        if not 0.0 <= self.horizontal_flip_augmentation_probability <= 1.0:
            raise ValueError(
                'horizontal_flip_augmentation_probability must be in [0, 1].'
            )
        if self.temporal_phase_offset < 0:
            raise ValueError('temporal_phase_offset must be non-negative.')
        if self.temporal_phase_offset >= self.temporal_bin_size:
            raise ValueError(
                'temporal_phase_offset must be smaller than temporal_bin_size.'
            )
        if (
            self.local_temporal_context_kernel_size <= 0
            or self.local_temporal_context_kernel_size % 2 == 0
        ):
            raise ValueError(
                'local_temporal_context_kernel_size must be a positive odd integer.'
            )

        if self.dense_only_enabled or self.low_density_only_enabled:
            filtered_paths = []
            for path in self.file_paths:
                with np.load(path) as events:
                    if event_count_matches_training_route(
                        np.asarray(events['ev_loc']).shape[0],
                        dense_only_enabled=self.dense_only_enabled,
                        dense_only_event_count_cutoff=(
                            self.dense_only_event_count_cutoff
                        ),
                        low_density_only_enabled=self.low_density_only_enabled,
                        low_density_only_event_count_cutoff=(
                            self.low_density_only_event_count_cutoff
                        ),
                    ):
                        filtered_paths.append(path)
            self.file_paths = filtered_paths
            if not self.file_paths:
                route_description = (
                    'event_count > {}'.format(self.dense_only_event_count_cutoff)
                    if self.dense_only_enabled else
                    'event_count <= {}'.format(
                        self.low_density_only_event_count_cutoff
                    )
                )
                raise RuntimeError(
                    'No training videos satisfy {}.'.format(route_description)
                )
        self.source_video_count = len(self.file_paths)

        self._videos = {}
        self._lru = OrderedDict()
        if self.cache_all_videos:
            for video_index in range(len(self.file_paths)):
                video = self._load_video(video_index)
                if self.sequence_length > len(video.event_indices_by_bin):
                    raise ValueError(
                        'sequence_length exceeds the available temporal bins.'
                )
                self._videos[video_index] = video

        self._fast_positive_bins_by_video = [
            None for _ in range(len(self.file_paths))
        ]
        if self.motion_sampling_enabled and self.cache_all_videos:
            for video_index in range(len(self.file_paths)):
                self._fast_positive_bins_by_video[video_index] = (
                    self._build_fast_positive_bins(
                        self._videos[video_index],
                        self.motion_sampling_min_displacement,
                    )
                )

        self.source_views_by_video = np.full(
            len(self.file_paths),
            self.views_per_video,
            dtype=np.int64,
        )
        dense_mask = np.zeros(len(self.file_paths), dtype=bool)
        mid_density_mask = np.zeros(len(self.file_paths), dtype=bool)
        event_counts = np.zeros(len(self.file_paths), dtype=np.int64)
        if (
            self.dense_sampling_enabled
            or self.mid_density_sampling_enabled
            or self.cross_video_copy_paste_enabled
        ):
            for video_index in range(len(self.file_paths)):
                event_counts[video_index] = self._video(
                    video_index
                ).locations.shape[0]
        if self.mid_density_sampling_enabled:
            mid_density_mask = (
                (event_counts > self.mid_density_min_event_count)
                & (event_counts <= self.mid_density_max_event_count)
            )
            self.source_views_by_video[mid_density_mask] *= (
                self.mid_density_view_multiplier
            )
        if self.dense_sampling_enabled:
            dense_mask = event_counts > self.dense_event_count_cutoff
            self.source_views_by_video[dense_mask] *= self.dense_view_multiplier
        self.mid_density_video_count = int(mid_density_mask.sum())
        self.extra_mid_density_views = int(
            self.views_per_video
            * (self.mid_density_view_multiplier - 1)
            * self.mid_density_video_count
        )
        self.dense_video_count = int(dense_mask.sum())
        self.extra_dense_views = int(
            self.views_per_video
            * (self.dense_view_multiplier - 1)
            * self.dense_video_count
        )
        self.copy_paste_base_views_by_video = self.source_views_by_video.copy()
        copy_paste_mask = np.zeros(len(self.file_paths), dtype=bool)
        if self.cross_video_copy_paste_enabled and self.cross_video_copy_paste_extra_views:
            copy_paste_mask = event_counts > self.cross_video_copy_paste_min_event_count
            self.source_views_by_video[copy_paste_mask] += (
                self.cross_video_copy_paste_extra_views
            )
        self.copy_paste_video_count = int(copy_paste_mask.sum())
        self.copy_paste_extra_views_total = int(
            self.cross_video_copy_paste_extra_views * self.copy_paste_video_count
        )
        self.active_source_indices = np.empty(0, dtype=np.int64)
        self.views_by_video = np.empty(0, dtype=np.int64)
        self.view_offsets = np.zeros(1, dtype=np.int64)
        self.active_video_count = 0
        self.fast_bin_count = 0
        self.motion_selected_views = 0
        self.trajectory_augmented_views = 0
        self.cross_video_copy_paste_views = 0
        self.horizontal_flip_views = 0
        self.set_epoch(0)

    @staticmethod
    def _build_fast_positive_bins(video, min_displacement):
        """Find bins with a large labelled target displacement to the next bin."""
        positive_mask = (video.labels > 0.5) & (video.target_ids > 0)
        positive_indices = np.flatnonzero(positive_mask)
        if positive_indices.size == 0:
            return np.empty(0, dtype=np.int64)

        centers_by_target = {}
        target_ids = np.unique(video.target_ids[positive_indices])
        for target_id in target_ids:
            target_indices = positive_indices[
                video.target_ids[positive_indices] == target_id
            ]
            bins = video.event_bins[target_indices]
            centers = {}
            for temporal_bin in np.unique(bins):
                bin_indices = target_indices[bins == temporal_bin]
                locations = video.locations[bin_indices, :2].astype(
                    np.float64,
                    copy=False,
                )
                centers[int(temporal_bin)] = locations.mean(axis=0)
            centers_by_target[int(target_id)] = centers

        fast_bins = set()
        for centers in centers_by_target.values():
            sorted_bins = sorted(centers)
            for previous_bin, next_bin in zip(sorted_bins, sorted_bins[1:]):
                if next_bin != previous_bin + 1:
                    continue
                displacement = np.linalg.norm(
                    centers[next_bin] - centers[previous_bin]
                )
                if displacement >= float(min_displacement):
                    fast_bins.add(previous_bin)
                    fast_bins.add(next_bin)
        return np.asarray(sorted(fast_bins), dtype=np.int64)

    def _fast_positive_bins_for_video(self, video_index, video):
        fast_bins = self._fast_positive_bins_by_video[video_index]
        if fast_bins is None:
            fast_bins = self._build_fast_positive_bins(
                video,
                self.motion_sampling_min_displacement,
            )
            self._fast_positive_bins_by_video[video_index] = fast_bins
        return fast_bins

    def _active_indices_for_epoch(self):
        source_indices = np.arange(self.source_video_count, dtype=np.int64)
        if (
            self.max_videos_per_epoch <= 0
            or self.max_videos_per_epoch >= self.source_video_count
        ):
            return source_indices

        batches_per_cycle = int(np.ceil(
            self.source_video_count / float(self.max_videos_per_epoch)
        ))
        cycle_index = self.current_epoch // batches_per_cycle
        batch_index = self.current_epoch % batches_per_cycle
        rng = np.random.default_rng(
            self.random_seed + 1000003 * cycle_index
        )
        permutation = rng.permutation(source_indices)
        start = batch_index * self.max_videos_per_epoch
        stop = min(start + self.max_videos_per_epoch, self.source_video_count)
        return permutation[start:stop]

    def _refresh_active_views(self):
        self.active_source_indices = self._active_indices_for_epoch()
        self.active_video_count = int(self.active_source_indices.size)
        self.views_by_video = self.source_views_by_video[
            self.active_source_indices
        ]
        if self.motion_sampling_enabled:
            self.fast_bin_count = int(
                sum(
                    self._fast_positive_bins_for_video(
                        int(video_index),
                        self._video(int(video_index)),
                    ).size
                    for video_index in self.active_source_indices
                )
            )
        else:
            self.fast_bin_count = 0
        self.motion_selected_views = 0
        self.trajectory_augmented_views = 0
        self.cross_video_copy_paste_views = 0
        self.horizontal_flip_views = 0
        self.view_offsets = np.concatenate((
            np.zeros(1, dtype=np.int64),
            np.cumsum(self.views_by_video, dtype=np.int64),
        ))

    def _load_video(self, video_index):
        video = load_temporal_frame_video(
            self.file_paths[video_index],
            self.temporal_bin_size,
            self.whole_t,
        )
        if self.temporal_phase_offset:
            video = temporal_phase_shift_temporal_frame_video(
                video,
                self.temporal_bin_size,
                self.temporal_phase_offset,
            )
        return video

    def _video(self, video_index):
        cached = self._videos.get(video_index)
        if cached is not None:
            return cached
        cached = self._lru.pop(video_index, None)
        if cached is not None:
            self._lru[video_index] = cached
            return cached
        cached = self._load_video(video_index)
        if self.sequence_length > len(cached.event_indices_by_bin):
            raise ValueError(
                'sequence_length exceeds the available temporal bins.'
            )
        self._lru[video_index] = cached
        while len(self._lru) > self.cache_video_count:
            self._lru.popitem(last=False)
        return cached

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)
        self._refresh_active_views()

    def __len__(self):
        return int(self.view_offsets[-1])

    def _sample_center_bin(self, video_index, view_index, video):
        seed = (
            self.random_seed
            + 1000003 * self.current_epoch
            + 1009 * video_index
            + view_index
        )
        rng = np.random.default_rng(seed)
        use_positive = (
            video.positive_bins.size > 0
            and rng.random() < self.positive_frame_probability
        )
        candidates = video.positive_bins if use_positive else video.occupied_bins
        if candidates.size == 0:
            raise RuntimeError('{} contains no valid event-time bins.'.format(video.name))
        is_extra_view = view_index >= self.views_per_video
        motion_eligible = (
            self.motion_sampling_enabled
            and video.locations.shape[0] > self.motion_sampling_min_event_count
            and (
                not self.motion_sampling_extra_views_only
                or is_extra_view
            )
        )
        if motion_eligible:
            fast_positive_bins = self._fast_positive_bins_for_video(
                video_index,
                video,
            )
            if (
                fast_positive_bins.size > 0
                and rng.random() < self.motion_sampling_probability
            ):
                self.motion_selected_views += 1
                return int(fast_positive_bins[rng.integers(fast_positive_bins.size)])
        return int(candidates[rng.integers(candidates.size)])

    def _use_horizontal_flip(self, video_index, view_index):
        if not self.horizontal_flip_augmentation_enabled:
            return False
        seed = (
            self.random_seed
            + 2000003 * self.current_epoch
            + 2029 * int(video_index)
            + int(view_index)
        )
        return bool(
            np.random.default_rng(seed).random()
            < self.horizontal_flip_augmentation_probability
        )

    def _copy_paste_video(self, source_video_index, start_bin, view_index, video):
        eligible = (
            self.cross_video_copy_paste_enabled
            and video.locations.shape[0] > self.cross_video_copy_paste_min_event_count
            and (
                not self.cross_video_copy_paste_extra_views_only
                or view_index >= self.copy_paste_base_views_by_video[source_video_index]
            )
        )
        if not eligible or self.source_video_count < 2:
            return None
        seed = (
            self.random_seed
            + 5000021 * self.current_epoch
            + 4013 * int(source_video_index)
            + int(view_index)
        )
        rng = np.random.default_rng(seed)
        if rng.random() >= self.cross_video_copy_paste_probability:
            return None
        donor_indices = np.asarray(
            [
                donor_index
                for donor_index in range(self.source_video_count)
                if (
                    donor_index != int(source_video_index)
                    and self._video(donor_index).locations.shape[0]
                    > self.cross_video_copy_paste_min_event_count
                )
            ],
            dtype=np.int64,
        )
        if donor_indices.size == 0:
            return None
        for donor_index in rng.permutation(donor_indices).tolist():
            donor_video = self._video(int(donor_index))
            pasted = copy_paste_target_track(
                video,
                donor_video,
                start_bin,
                self.sequence_length,
                self.context_bins,
                self.temporal_bin_size,
                self.width,
                self.height,
                rng,
                min_track_bins=self.cross_video_copy_paste_min_track_bins,
                collision_radius=self.cross_video_copy_paste_collision_radius,
            )
            if pasted is not None:
                self.cross_video_copy_paste_views += 1
                return pasted
        return None

    def __getitem__(self, index):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError('Temporal-memory sample index is out of range.')
        active_video_index = int(
            np.searchsorted(self.view_offsets, index, side='right') - 1
        )
        source_video_index = int(self.active_source_indices[active_video_index])
        view_index = int(index - self.view_offsets[active_video_index])
        video = self._video(source_video_index)
        center_bin = self._sample_center_bin(
            source_video_index,
            view_index,
            video,
        )
        start_bin = temporal_sequence_start(
            center_bin,
            len(video.event_indices_by_bin),
            self.sequence_length,
        )

        sampling_video = video
        copy_paste_video = self._copy_paste_video(
            source_video_index,
            start_bin,
            view_index,
            video,
        )
        if copy_paste_video is not None:
            sampling_video = copy_paste_video
        trajectory_eligible = (
            self.trajectory_augmentation_enabled
            and video.locations.shape[0]
            > self.trajectory_augmentation_min_event_count
            and (
                not self.trajectory_augmentation_extra_views_only
                or view_index >= self.views_per_video
            )
        )
        if trajectory_eligible and sampling_video is video:
            augmentation_seed = (
                self.random_seed
                + 3000017 * self.current_epoch
                + 3079 * source_video_index
                + view_index
            )
            augmentation_rng = np.random.default_rng(augmentation_seed)
            if (
                augmentation_rng.random()
                < self.trajectory_augmentation_probability
            ):
                augmented_video = augment_target_trajectory(
                    video,
                    start_bin,
                    self.sequence_length,
                    self.context_bins,
                    self.temporal_bin_size,
                    self.width,
                    self.height,
                    self.trajectory_augmentation_residual_speed,
                    augmentation_rng,
                    self.trajectory_augmentation_min_track_bins,
                )
                if augmented_video is not None:
                    sampling_video = augmented_video
                    self.trajectory_augmented_views += 1

        frames = []
        event_time_indices = []
        event_times = []
        event_x = []
        event_y = []
        labels = []
        target_ids = []
        for sequence_index, temporal_bin in enumerate(
            range(start_bin, start_bin + self.sequence_length)
        ):
            frames.append(
                build_temporal_context_frame(
                    sampling_video,
                    temporal_bin,
                    self.context_bins,
                    self.width,
                    self.height,
                    self.log_count_clip,
                    local_temporal_context_enabled=(
                        self.local_temporal_context_enabled
                    ),
                    local_temporal_context_kernel_size=(
                        self.local_temporal_context_kernel_size
                    ),
                )
            )
            event_indices = sampling_video.event_indices_by_bin[temporal_bin]
            if event_indices.size == 0:
                continue
            locations = sampling_video.locations[event_indices]
            event_time_indices.append(
                np.full(event_indices.shape, sequence_index, dtype=np.int64)
            )
            event_times.append(locations[:, 2].astype(np.int64, copy=False))
            event_x.append(locations[:, 0].astype(np.int64, copy=False))
            event_y.append(locations[:, 1].astype(np.int64, copy=False))
            labels.append(sampling_video.labels[event_indices].astype(np.float32, copy=False))
            target_ids.append(
                sampling_video.target_ids[event_indices].astype(np.int64, copy=False)
            )

        if not event_time_indices:
            raise RuntimeError('Sampled sequence contains no events.')
        frames = np.stack(frames, axis=0)
        event_x = np.concatenate(event_x)
        if self._use_horizontal_flip(source_video_index, view_index):
            frames, event_x = horizontal_flip_temporal_memory_sequence(
                frames,
                event_x,
                self.width,
            )
            self.horizontal_flip_views += 1
        return {
            'frames': frames,
            'event_time_indices': np.concatenate(event_time_indices),
            # Preserve raw timestamps for training-only losses tied to the
            # official 50-unit Pd windows. Local sequence indices are not a
            # substitute because each sampled view starts at a different bin.
            'event_times': np.concatenate(event_times),
            'event_x': event_x,
            'event_y': np.concatenate(event_y),
            'labels': np.concatenate(labels),
            'target_ids': np.concatenate(target_ids),
        }


def temporal_memory_collate(samples):
    """Keep one variable-event sequence per GPU step for predictable memory."""
    if len(samples) != 1:
        raise ValueError('Temporal-memory training requires batch_size=1.')
    sample = samples[0]
    return {
        'frames': torch.from_numpy(sample['frames']).float(),
        'event_time_indices': torch.from_numpy(
            sample['event_time_indices']
        ).long(),
        'event_times': torch.from_numpy(sample['event_times']).long(),
        'event_x': torch.from_numpy(sample['event_x']).long(),
        'event_y': torch.from_numpy(sample['event_y']).long(),
        'labels': torch.from_numpy(sample['labels']).float(),
        'target_ids': torch.from_numpy(sample['target_ids']).long(),
    }
