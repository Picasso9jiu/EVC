"""Full-stream temporal event-frame views for the auxiliary 2D model.

The sparse baseline receives a budgeted event cloud. This module instead
builds a small stack of polarity-count images around one 50-unit time bin, so
every event remains observable at training and inference time.
"""

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def temporal_bin_count(whole_t, temporal_bin_size):
    whole_t = int(whole_t)
    temporal_bin_size = int(temporal_bin_size)
    if whole_t <= 0 or temporal_bin_size <= 0:
        raise ValueError('whole_t and temporal_bin_size must be positive.')
    return (whole_t + temporal_bin_size - 1) // temporal_bin_size


def temporal_event_bins(locations, temporal_bin_size, bin_count):
    """Return one clipped temporal-bin index for every event."""
    locations = np.asarray(locations)
    if locations.ndim != 2 or locations.shape[1] < 3:
        raise ValueError('locations must have shape [N, 3+] ordered as x, y, t.')
    temporal_bin_size = int(temporal_bin_size)
    bin_count = int(bin_count)
    if temporal_bin_size <= 0 or bin_count <= 0:
        raise ValueError('temporal_bin_size and bin_count must be positive.')
    timestamps = locations[:, 2].astype(np.int64, copy=False)
    return np.clip(timestamps // temporal_bin_size, 0, bin_count - 1)


@dataclass
class TemporalFrameVideo:
    """One complete labelled video held in host memory."""

    name: str
    locations: np.ndarray
    polarities: np.ndarray
    labels: np.ndarray
    target_ids: np.ndarray
    event_bins: np.ndarray
    event_indices_by_bin: tuple
    positive_bins: np.ndarray
    occupied_bins: np.ndarray


def temporal_frame_video_from_events(
    name,
    locations,
    polarities,
    temporal_bin_size,
    whole_t,
    labels=None,
    target_ids=None,
):
    """Index raw event inputs without requiring labels at inference time."""
    locations = np.asarray(locations)
    polarities = np.asarray(polarities).reshape(-1)
    if locations.ndim != 2 or locations.shape[1] < 3:
        raise ValueError('locations must have shape [N, 3+] ordered as x, y, t.')
    if locations.shape[0] != polarities.shape[0]:
        raise ValueError('locations and polarities must have matching lengths.')
    event_count = locations.shape[0]
    if labels is None:
        labels = np.zeros(event_count, dtype=np.float32)
    if target_ids is None:
        target_ids = np.zeros(event_count, dtype=np.int64)
    labels = np.asarray(labels).reshape(-1).astype(np.float32, copy=False)
    target_ids = np.asarray(target_ids).reshape(-1).astype(np.int64, copy=False)
    if labels.shape[0] != event_count or target_ids.shape[0] != event_count:
        raise ValueError('Event labels and target ids must match locations.')
    bin_count = temporal_bin_count(whole_t, temporal_bin_size)
    event_bins = temporal_event_bins(locations, temporal_bin_size, bin_count)
    event_indices_by_bin = tuple(
        np.flatnonzero(event_bins == temporal_bin).astype(np.int64, copy=False)
        for temporal_bin in range(bin_count)
    )
    return TemporalFrameVideo(
        name=str(name),
        locations=locations[:, :3].astype(np.int64, copy=False),
        polarities=polarities.astype(np.float32, copy=False),
        labels=labels,
        target_ids=target_ids,
        event_bins=event_bins,
        event_indices_by_bin=event_indices_by_bin,
        positive_bins=np.flatnonzero(
            np.bincount(
                event_bins[labels > 0.5],
                minlength=bin_count,
            )
            > 0
        ).astype(np.int64, copy=False),
        occupied_bins=np.flatnonzero(
            np.bincount(event_bins, minlength=bin_count) > 0
        ).astype(np.int64, copy=False),
    )


def temporal_phase_shift_temporal_frame_video(
    video,
    temporal_bin_size,
    phase_offset,
):
    """Re-index a full stream on a nonzero intra-bin time phase.

    Event order, coordinates, labels, and target IDs remain aligned. Only the
    time coordinate used for frame formation is translated. The added tail
    bin intentionally matches P41 inference, so a phase expert trains on the
    same input view it receives at evaluation time.
    """
    if not isinstance(video, TemporalFrameVideo):
        raise TypeError('video must be a TemporalFrameVideo.')
    temporal_bin_size = int(temporal_bin_size)
    phase_offset = int(phase_offset)
    if temporal_bin_size <= 0:
        raise ValueError('temporal_bin_size must be positive.')
    if phase_offset <= 0 or phase_offset >= temporal_bin_size:
        raise ValueError('phase_offset must be in [1, temporal_bin_size - 1].')
    shifted_locations = video.locations.copy()
    shifted_locations[:, 2] += phase_offset
    shifted_whole_t = temporal_bin_size * len(video.event_indices_by_bin) + phase_offset
    if shifted_locations.size:
        shifted_whole_t = max(
            shifted_whole_t,
            int(shifted_locations[:, 2].max()) + 1,
        )
    return temporal_frame_video_from_events(
        name=video.name,
        locations=shifted_locations,
        polarities=video.polarities,
        temporal_bin_size=temporal_bin_size,
        whole_t=shifted_whole_t,
        labels=video.labels,
        target_ids=video.target_ids,
    )


def load_temporal_frame_video(path, temporal_bin_size, whole_t):
    """Load an EV-UAV npz file and index its events by metric-time bins."""
    path = Path(path)
    with np.load(path) as events:
        evs_norm = np.asarray(events['evs_norm'])
        locations = np.asarray(events['ev_loc'])
    if evs_norm.ndim != 2 or evs_norm.shape[1] < 6:
        raise ValueError('{}: evs_norm must have at least six columns.'.format(path))
    return temporal_frame_video_from_events(
        name=path.stem,
        locations=locations,
        polarities=evs_norm[:, 3],
        temporal_bin_size=temporal_bin_size,
        whole_t=whole_t,
        labels=evs_norm[:, 4],
        target_ids=evs_norm[:, 5],
    )


def build_temporal_context_frame(
    video,
    center_bin,
    context_bins,
    width,
    height,
    log_count_clip=4.0,
    local_contrast_enabled=False,
    local_contrast_kernel_size=9,
    local_temporal_context_enabled=False,
    local_temporal_context_kernel_size=11,
):
    """Build a normalized polarity-count frame stack centred on one time bin.

    Context bins must be odd. For five bins the output channels are
    negative/positive counts for center-2 through center+2.
    """
    if not isinstance(video, TemporalFrameVideo):
        raise TypeError('video must be a TemporalFrameVideo.')
    context_bins = int(context_bins)
    width = int(width)
    height = int(height)
    center_bin = int(center_bin)
    log_count_clip = float(log_count_clip)
    if context_bins < 1 or context_bins % 2 == 0:
        raise ValueError('context_bins must be a positive odd integer.')
    if width <= 0 or height <= 0:
        raise ValueError('width and height must be positive.')
    if log_count_clip <= 0:
        raise ValueError('log_count_clip must be positive.')
    if center_bin < 0 or center_bin >= len(video.event_indices_by_bin):
        raise ValueError('center_bin is outside the video temporal range.')

    frame = np.zeros((context_bins * 2, height, width), dtype=np.float32)
    half_context = context_bins // 2
    pixel_count = height * width
    for relative_bin in range(-half_context, half_context + 1):
        source_bin = center_bin + relative_bin
        if source_bin < 0 or source_bin >= len(video.event_indices_by_bin):
            continue
        event_indices = video.event_indices_by_bin[source_bin]
        if event_indices.size == 0:
            continue
        coordinates = video.locations[event_indices]
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        valid = (
            (x >= 0)
            & (x < width)
            & (y >= 0)
            & (y < height)
        )
        if not valid.any():
            continue
        polarity = (
            video.polarities[event_indices][valid] > 0.5
        ).astype(np.int64, copy=False)
        channel = (relative_bin + half_context) * 2 + polarity
        flat_indices = (
            channel * pixel_count
            + y[valid].astype(np.int64, copy=False) * width
            + x[valid].astype(np.int64, copy=False)
        )
        np.add.at(frame.reshape(-1), flat_indices, 1.0)

    local_temporal_context = None
    if local_temporal_context_enabled:
        valid_neighbour_bins = sum(
            1
            for relative_bin in range(-half_context, half_context + 1)
            if relative_bin != 0
            and 0 <= center_bin + relative_bin < len(video.event_indices_by_bin)
        )
        local_temporal_context = build_local_temporal_context_channel(
            frame,
            context_bins,
            local_temporal_context_kernel_size,
            log_count_clip,
            valid_neighbour_bins=valid_neighbour_bins,
        )

    np.log1p(frame, out=frame)
    np.minimum(frame, log_count_clip, out=frame)
    frame /= log_count_clip
    if local_contrast_enabled:
        frame = append_local_density_contrast(
            frame,
            local_contrast_kernel_size,
        )
    if local_temporal_context is not None:
        frame = np.concatenate((frame, local_temporal_context), axis=0)
    return frame


def _local_box_mean(frame, kernel_size):
    """Return an edge-padded per-channel box mean without external deps."""
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 3:
        raise ValueError('frame must have shape [C, H, W].')
    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError('local_contrast_kernel_size must be a positive odd integer.')
    if frame.shape[1] <= 0 or frame.shape[2] <= 0:
        raise ValueError('frame spatial dimensions must be positive.')

    radius = kernel_size // 2
    padded = np.pad(
        frame,
        ((0, 0), (radius, radius), (radius, radius)),
        mode='edge',
    )
    integral = np.pad(
        np.cumsum(np.cumsum(padded, axis=1), axis=2),
        ((0, 0), (1, 0), (1, 0)),
        mode='constant',
    )
    box_sum = (
        integral[:, kernel_size:, kernel_size:]
        - integral[:, :-kernel_size, kernel_size:]
        - integral[:, kernel_size:, :-kernel_size]
        + integral[:, :-kernel_size, :-kernel_size]
    )
    return box_sum / float(kernel_size * kernel_size)


def append_local_density_contrast(frame, kernel_size=9):
    """Append local count contrast while preserving the original frame channels."""
    frame = np.asarray(frame, dtype=np.float32)
    local_mean = _local_box_mean(frame, kernel_size)
    return np.concatenate((frame, frame - local_mean), axis=0)


def build_local_temporal_context_channel(
    raw_frame,
    context_bins,
    kernel_size=11,
    log_count_clip=4.0,
    valid_neighbour_bins=None,
):
    """Return local activity from the non-centre temporal context bins.

    The main event frame preserves its existing polarity-count inputs.  This
    optional one-channel feature instead measures how much *nearby* activity
    occurs at the same location in the two preceding and two following bins.
    It is built before log clipping so a persistent high-rate background is
    distinguishable even when the centre-bin event count is identical.
    """
    raw_frame = np.asarray(raw_frame, dtype=np.float32)
    context_bins = int(context_bins)
    kernel_size = int(kernel_size)
    log_count_clip = float(log_count_clip)
    if raw_frame.ndim != 3 or raw_frame.shape[0] != context_bins * 2:
        raise ValueError(
            'raw_frame must have shape [context_bins * 2, H, W].'
        )
    if context_bins < 3 or context_bins % 2 == 0:
        raise ValueError('context_bins must be an odd integer of at least three.')
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            'local_temporal_context_kernel_size must be a positive odd integer.'
        )
    if log_count_clip <= 0.0:
        raise ValueError('log_count_clip must be positive.')

    centre_start = (context_bins // 2) * 2
    neighbour_activity = raw_frame.sum(axis=0, keepdims=True).copy()
    neighbour_activity -= raw_frame[centre_start:centre_start + 2].sum(
        axis=0,
        keepdims=True,
    )
    if valid_neighbour_bins is None:
        valid_neighbour_bins = context_bins - 1
    valid_neighbour_bins = int(valid_neighbour_bins)
    if valid_neighbour_bins <= 0 or valid_neighbour_bins > context_bins - 1:
        raise ValueError(
            'valid_neighbour_bins must be in [1, context_bins - 1].'
        )
    # M71 measures the mean activity over the *available* neighbouring bins.
    # Constant padding keeps the spatial border consistent with that diagnostic:
    # out-of-image pixels are absent rather than replicated.
    radius = kernel_size // 2
    padded = np.pad(
        neighbour_activity,
        ((0, 0), (radius, radius), (radius, radius)),
        mode='constant',
    )
    integral = np.pad(
        np.cumsum(np.cumsum(padded, axis=1), axis=2),
        ((0, 0), (1, 0), (1, 0)),
        mode='constant',
    )
    local_sum = (
        integral[:, kernel_size:, kernel_size:]
        - integral[:, :-kernel_size, kernel_size:]
        - integral[:, kernel_size:, :-kernel_size]
        + integral[:, :-kernel_size, :-kernel_size]
    )
    local_sum /= float(valid_neighbour_bins)
    np.log1p(local_sum, out=local_sum)
    np.minimum(local_sum, log_count_clip, out=local_sum)
    return local_sum / log_count_clip


def temporal_frame_view_schedule(
    event_counts,
    views_per_video,
    dense_sampling_enabled=False,
    dense_event_count_cutoff=200000,
    dense_view_multiplier=2,
):
    """Build a deterministic view schedule with optional dense-video repeats."""
    event_counts = np.asarray(event_counts, dtype=np.int64).reshape(-1)
    views_per_video = int(views_per_video)
    dense_event_count_cutoff = int(dense_event_count_cutoff)
    dense_view_multiplier = int(dense_view_multiplier)
    if views_per_video <= 0:
        raise ValueError('views_per_video must be positive.')
    if dense_event_count_cutoff <= 0:
        raise ValueError('dense_event_count_cutoff must be positive.')
    if dense_view_multiplier < 1:
        raise ValueError('dense_view_multiplier must be at least one.')
    if np.any(event_counts < 0):
        raise ValueError('event_counts must not contain negative values.')

    schedule = []
    for video_index, event_count in enumerate(event_counts.tolist()):
        view_count = views_per_video
        if (
            dense_sampling_enabled
            and event_count >= dense_event_count_cutoff
        ):
            view_count *= dense_view_multiplier
        schedule.extend(
            (video_index, view_index)
            for view_index in range(view_count)
        )
    return tuple(schedule)


class TemporalFrameTrainDataset(Dataset):
    """Random, deterministic metric-time views sampled from full videos."""

    def __init__(
        self,
        root,
        whole_t,
        temporal_bin_size,
        context_bins,
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
        fine_detail_enabled=False,
        fine_temporal_bin_size=25,
        fine_context_bins=9,
    ):
        self.root = Path(root)
        self.whole_t = int(whole_t)
        self.temporal_bin_size = int(temporal_bin_size)
        self.context_bins = int(context_bins)
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
        self.fine_detail_enabled = bool(fine_detail_enabled)
        self.fine_temporal_bin_size = int(fine_temporal_bin_size)
        self.fine_context_bins = int(fine_context_bins)
        self.current_epoch = 0
        self.file_paths = sorted(self.root.glob('*.npz'))
        if not self.file_paths:
            raise RuntimeError('No npz files found in {}'.format(self.root))
        if self.views_per_video <= 0:
            raise ValueError('views_per_video must be positive.')
        if not 0.0 <= self.positive_frame_probability <= 1.0:
            raise ValueError('positive_frame_probability must be in [0, 1].')
        if self.cache_video_count <= 0:
            raise ValueError('cache_video_count must be positive.')
        if self.context_bins < 1 or self.context_bins % 2 == 0:
            raise ValueError('context_bins must be a positive odd integer.')
        if self.dense_event_count_cutoff <= 0:
            raise ValueError('dense_event_count_cutoff must be positive.')
        if self.dense_view_multiplier < 1:
            raise ValueError('dense_view_multiplier must be at least one.')
        if self.fine_detail_enabled:
            if self.fine_temporal_bin_size <= 0:
                raise ValueError('fine_temporal_bin_size must be positive.')
            if self.fine_temporal_bin_size > self.temporal_bin_size:
                raise ValueError(
                    'fine_temporal_bin_size must not exceed '
                    'temporal_bin_size.'
                )
            if self.temporal_bin_size % self.fine_temporal_bin_size != 0:
                raise ValueError(
                    'temporal_bin_size must be divisible by '
                    'fine_temporal_bin_size.'
                )
            if self.fine_context_bins < 1 or self.fine_context_bins % 2 == 0:
                raise ValueError(
                    'fine_context_bins must be a positive odd integer.'
                )
            self.fine_bins_per_coarse = (
                self.temporal_bin_size // self.fine_temporal_bin_size
            )
        else:
            self.fine_bins_per_coarse = 1

        self._videos = {}
        self._lru = OrderedDict()
        self._fine_videos = {}
        self._fine_lru = OrderedDict()
        if self.cache_all_videos:
            for video_index in range(len(self.file_paths)):
                video = self._load_video(video_index)
                self._videos[video_index] = video
                if self.fine_detail_enabled:
                    self._fine_videos[video_index] = self._build_fine_video(video)
        if self.dense_sampling_enabled:
            event_counts = [
                self._video(video_index).locations.shape[0]
                for video_index in range(len(self.file_paths))
            ]
        else:
            event_counts = [0] * len(self.file_paths)
        self._view_schedule = temporal_frame_view_schedule(
            event_counts,
            self.views_per_video,
            self.dense_sampling_enabled,
            self.dense_event_count_cutoff,
            self.dense_view_multiplier,
        )
        self.dense_video_count = (
            sum(
                event_count >= self.dense_event_count_cutoff
                for event_count in event_counts
            )
            if self.dense_sampling_enabled else 0
        )

    def _load_video(self, video_index):
        return load_temporal_frame_video(
            self.file_paths[video_index],
            self.temporal_bin_size,
            self.whole_t,
        )

    def _video(self, video_index):
        cached = self._videos.get(video_index)
        if cached is not None:
            return cached
        cached = self._lru.pop(video_index, None)
        if cached is not None:
            self._lru[video_index] = cached
            return cached
        cached = self._load_video(video_index)
        self._lru[video_index] = cached
        while len(self._lru) > self.cache_video_count:
            self._lru.popitem(last=False)
        return cached

    def _build_fine_video(self, video):
        return temporal_frame_video_from_events(
            name=video.name,
            locations=video.locations,
            polarities=video.polarities,
            temporal_bin_size=self.fine_temporal_bin_size,
            whole_t=self.whole_t,
            labels=video.labels,
            target_ids=video.target_ids,
        )

    def _fine_video(self, video_index, video):
        if not self.fine_detail_enabled:
            return None
        cached = self._fine_videos.get(video_index)
        if cached is not None:
            return cached
        cached = self._fine_lru.pop(video_index, None)
        if cached is not None:
            self._fine_lru[video_index] = cached
            return cached
        cached = self._build_fine_video(video)
        self._fine_lru[video_index] = cached
        while len(self._fine_lru) > self.cache_video_count:
            self._fine_lru.popitem(last=False)
        return cached

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def __len__(self):
        return len(self._view_schedule)

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
        return int(candidates[rng.integers(candidates.size)])

    def __getitem__(self, index):
        video_index, view_index = self._view_schedule[int(index)]
        video = self._video(video_index)
        fine_video = self._fine_video(video_index, video)
        sampling_video = fine_video if fine_video is not None else video
        center_bin = self._sample_center_bin(
            video_index,
            view_index,
            sampling_video,
        )
        event_indices = sampling_video.event_indices_by_bin[center_bin]
        if event_indices.size == 0:
            raise RuntimeError('Sampled an empty event-time bin.')
        coarse_center_bin = (
            center_bin // self.fine_bins_per_coarse
            if fine_video is not None else center_bin
        )
        frame = build_temporal_context_frame(
            video,
            coarse_center_bin,
            self.context_bins,
            self.width,
            self.height,
            self.log_count_clip,
        )
        locations = video.locations[event_indices]
        sample = {
            'frame': frame,
            'event_x': locations[:, 0].astype(np.int64, copy=False),
            'event_y': locations[:, 1].astype(np.int64, copy=False),
            'labels': video.labels[event_indices].astype(np.float32, copy=False),
            'target_ids': video.target_ids[event_indices].astype(
                np.int64,
                copy=False,
            ),
            'video_index': video_index,
            'center_bin': coarse_center_bin,
        }
        if fine_video is not None:
            sample['fine_detail_frame'] = build_temporal_context_frame(
                fine_video,
                center_bin,
                self.fine_context_bins,
                self.width,
                self.height,
                self.log_count_clip,
            )
        return sample


def temporal_frame_collate(samples):
    """Collate fixed-size image stacks and variable-size event labels."""
    if not samples:
        raise ValueError('Cannot collate an empty sample list.')
    frames = np.stack([sample['frame'] for sample in samples], axis=0)
    fine_detail_flags = [
        'fine_detail_frame' in sample for sample in samples
    ]
    if any(fine_detail_flags) and not all(fine_detail_flags):
        raise ValueError(
            'fine_detail_frame must be present for every sample or none.'
        )
    fine_detail_frames = (
        np.stack([sample['fine_detail_frame'] for sample in samples], axis=0)
        if all(fine_detail_flags) else None
    )
    video_indices = []
    center_bins = []
    event_x = []
    event_y = []
    labels = []
    target_ids = []
    event_batch_indices = []
    for batch_index, sample in enumerate(samples):
        video_indices.append(sample.get('video_index', 0))
        center_bins.append(sample.get('center_bin', 0))
        sample_x = np.asarray(sample['event_x'], dtype=np.int64)
        sample_y = np.asarray(sample['event_y'], dtype=np.int64)
        sample_labels = np.asarray(sample['labels'], dtype=np.float32)
        sample_target_ids = np.asarray(
            sample.get('target_ids', np.zeros_like(sample_labels)),
            dtype=np.int64,
        )
        if not (
            sample_x.shape
            == sample_y.shape
            == sample_labels.shape
            == sample_target_ids.shape
        ):
            raise ValueError('Sample event coordinates and labels must match.')
        event_x.append(sample_x)
        event_y.append(sample_y)
        labels.append(sample_labels)
        target_ids.append(sample_target_ids)
        event_batch_indices.append(
            np.full(sample_labels.shape, batch_index, dtype=np.int64)
        )
    batch = {
        'frames': torch.from_numpy(frames).float(),
        'video_indices': torch.tensor(video_indices, dtype=torch.long),
        'center_bins': torch.tensor(center_bins, dtype=torch.long),
        'event_x': torch.from_numpy(np.concatenate(event_x)).long(),
        'event_y': torch.from_numpy(np.concatenate(event_y)).long(),
        'labels': torch.from_numpy(np.concatenate(labels)).float(),
        'target_ids': torch.from_numpy(np.concatenate(target_ids)).long(),
        'event_batch_indices': torch.from_numpy(
            np.concatenate(event_batch_indices)
        ).long(),
    }
    if fine_detail_frames is not None:
        batch['fine_detail_frames'] = torch.from_numpy(
            fine_detail_frames
        ).float()
    return batch
