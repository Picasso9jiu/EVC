"""Training-only cross-video target event-track copy-paste augmentation."""

from __future__ import annotations

import numpy as np

from dataset.temporal_frame import (
    TemporalFrameVideo,
    temporal_frame_video_from_events,
)


def _track_candidates(video, first_bin, last_bin, min_track_bins):
    labels = video.labels > 0.5
    target_ids = video.target_ids.astype(np.int64, copy=False)
    event_bins = video.event_bins.astype(np.int64, copy=False)
    support = (event_bins >= int(first_bin)) & (event_bins <= int(last_bin))
    candidates = []
    for target_id in np.unique(
        target_ids[labels & support & (target_ids > 0)]
    ):
        target_mask = labels & support & (target_ids == int(target_id))
        bins = np.unique(event_bins[target_mask])
        if bins.size < int(min_track_bins):
            continue
        adjacent = sum(
            int(current == previous + 1)
            for previous, current in zip(bins[:-1], bins[1:])
        )
        if adjacent < int(min_track_bins) - 1:
            continue
        points = video.locations[target_mask, :2].astype(
            np.int64, copy=False
        )
        if points.size == 0:
            continue
        candidates.append((int(target_id), target_mask, points))
    return candidates


def _collision_free(shifted_points, base_positive_points, radius):
    if base_positive_points.size == 0:
        return True
    occupied = {tuple(point) for point in base_positive_points.tolist()}
    radius = int(radius)
    for x, y in shifted_points.tolist():
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if (int(x) + dx, int(y) + dy) in occupied:
                    return False
    return True


def copy_paste_target_track(
    base_video,
    donor_video,
    base_sequence_start,
    sequence_length,
    context_bins,
    temporal_bin_size,
    width,
    height,
    rng,
    min_track_bins=3,
    collision_radius=1,
    max_attempts=24,
):
    """Paste one donor target track into a base sequence, or return ``None``.

    Donor timestamps are remapped by temporal-bin offset while preserving
    their intra-bin phase.  Only donor events belonging to the selected target
    ID are appended; base events, labels and polarities remain untouched.
    """
    if not isinstance(base_video, TemporalFrameVideo):
        raise TypeError('base_video must be a TemporalFrameVideo.')
    if not isinstance(donor_video, TemporalFrameVideo):
        raise TypeError('donor_video must be a TemporalFrameVideo.')
    sequence_length = int(sequence_length)
    context_bins = int(context_bins)
    temporal_bin_size = int(temporal_bin_size)
    width = int(width)
    height = int(height)
    base_sequence_start = int(base_sequence_start)
    if sequence_length <= 1 or context_bins <= 0 or context_bins % 2 != 1:
        raise ValueError('sequence_length/context_bins are invalid.')
    if temporal_bin_size <= 0 or width <= 0 or height <= 0:
        raise ValueError('temporal_bin_size/geometry are invalid.')
    if not hasattr(rng, 'integers'):
        raise TypeError('rng must be a NumPy generator.')

    half_context = context_bins // 2
    base_first = max(0, base_sequence_start - half_context)
    base_last = min(
        len(base_video.event_indices_by_bin) - 1,
        base_sequence_start + sequence_length - 1 + half_context,
    )
    donor_bin_count = len(donor_video.event_indices_by_bin)
    donor_starts = np.arange(
        0,
        max(0, donor_bin_count - sequence_length + 1),
        dtype=np.int64,
    )
    if donor_starts.size == 0:
        return None
    donor_start_order = rng.permutation(donor_starts)

    base_event_bins = base_video.event_bins.astype(np.int64, copy=False)
    base_positive = (
        (base_video.labels > 0.5)
        & (base_video.target_ids > 0)
        & (base_event_bins >= base_first)
        & (base_event_bins <= base_last)
    )
    base_positive_points = base_video.locations[base_positive, :2].astype(
        np.int64, copy=False
    )

    for donor_start in donor_start_order.tolist():
        donor_first = max(0, int(donor_start) - half_context)
        donor_last = min(
            donor_bin_count - 1,
            int(donor_start) + sequence_length - 1 + half_context,
        )
        candidates = _track_candidates(
            donor_video,
            donor_first,
            donor_last,
            min_track_bins,
        )
        if not candidates:
            continue
        target_id, target_mask, points = candidates[
            int(rng.integers(len(candidates)))
        ]
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        min_dx, max_dx = -int(x0), width - 1 - int(x1)
        min_dy, max_dy = -int(y0), height - 1 - int(y1)
        if min_dx > max_dx or min_dy > max_dy:
            continue

        target_locations = donor_video.locations[target_mask].copy()
        target_bins = donor_video.event_bins[target_mask].astype(
            np.int64, copy=False
        )
        target_offsets = target_locations[:, 2] - target_bins * temporal_bin_size
        for _ in range(int(max_attempts)):
            shift = np.asarray((
                int(rng.integers(min_dx, max_dx + 1)),
                int(rng.integers(min_dy, max_dy + 1)),
            ), dtype=np.int64)
            shifted_xy = target_locations[:, :2] + shift[None, :]
            if not _collision_free(
                shifted_xy,
                base_positive_points,
                collision_radius,
            ):
                continue
            if np.array_equal(shifted_xy, target_locations[:, :2]):
                continue

            mapped_bins = base_first + (target_bins - donor_first)
            if (
                (mapped_bins < 0).any()
                or (mapped_bins >= len(base_video.event_indices_by_bin)).any()
            ):
                continue
            pasted_locations = np.column_stack((
                shifted_xy,
                mapped_bins * temporal_bin_size + target_offsets,
            )).astype(np.int64, copy=False)
            pasted_polarities = donor_video.polarities[target_mask].copy()
            pasted_labels = np.ones(pasted_locations.shape[0], dtype=np.float32)
            existing_ids = base_video.target_ids.astype(np.int64, copy=False)
            pasted_target_id = int(existing_ids.max()) + 1 if existing_ids.size else 1
            pasted_ids = np.full(
                pasted_locations.shape[0],
                pasted_target_id,
                dtype=np.int64,
            )
            locations = np.concatenate((base_video.locations, pasted_locations), axis=0)
            polarities = np.concatenate((base_video.polarities, pasted_polarities), axis=0)
            labels = np.concatenate((base_video.labels, pasted_labels), axis=0)
            target_ids = np.concatenate((base_video.target_ids, pasted_ids), axis=0)
            whole_t = max(
                int(len(base_video.event_indices_by_bin) * temporal_bin_size),
                int(locations[:, 2].max()) + 1 if locations.size else 1,
            )
            return temporal_frame_video_from_events(
                name=base_video.name + '__m93_copy_paste',
                locations=locations,
                polarities=polarities,
                temporal_bin_size=temporal_bin_size,
                whole_t=whole_t,
                labels=labels,
                target_ids=target_ids,
            )
    return None
