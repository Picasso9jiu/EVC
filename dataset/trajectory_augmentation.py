"""Label-preserving target trajectory augmentation for training only.

The augmentation adds a constant residual velocity to one labelled target
track while leaving every background event, timestamp, polarity, label and
target id unchanged.  It is deliberately conservative: a candidate that
would leave the sensor image is rejected instead of being clipped.
"""

from __future__ import annotations

import numpy as np

from dataset.temporal_frame import (
    TemporalFrameVideo,
    temporal_frame_video_from_events,
)


def augment_target_trajectory(
    video,
    sequence_start,
    sequence_length,
    context_bins,
    temporal_bin_size,
    width,
    height,
    residual_speed,
    rng,
    min_track_bins=3,
):
    """Return one synthetic faster-target view or ``None``.

    The residual vector follows the observed direction of the selected target
    when one exists.  It is applied around the sequence centre, so the target
    remains in roughly the same field of view and only its velocity changes.
    """
    if not isinstance(video, TemporalFrameVideo):
        raise TypeError('video must be a TemporalFrameVideo.')
    sequence_start = int(sequence_start)
    sequence_length = int(sequence_length)
    context_bins = int(context_bins)
    temporal_bin_size = int(temporal_bin_size)
    width = int(width)
    height = int(height)
    residual_speed = float(residual_speed)
    min_track_bins = int(min_track_bins)
    if sequence_start < 0 or sequence_length <= 0:
        raise ValueError('sequence_start and sequence_length are invalid.')
    if context_bins <= 0 or context_bins % 2 == 0 or temporal_bin_size <= 0:
        raise ValueError('context_bins must be a positive odd integer.')
    if width <= 0 or height <= 0 or residual_speed <= 0.0:
        raise ValueError('width, height and residual_speed must be positive.')
    if min_track_bins < 2:
        raise ValueError('min_track_bins must be at least two.')
    if not hasattr(rng, 'choice'):
        raise TypeError('rng must be a NumPy generator.')

    half_context = context_bins // 2
    first_bin = max(0, sequence_start - half_context)
    last_bin = min(
        len(video.event_indices_by_bin) - 1,
        sequence_start + sequence_length - 1 + half_context,
    )
    if first_bin > last_bin:
        return None

    labels = video.labels > 0.5
    target_ids = video.target_ids.astype(np.int64, copy=False)
    event_bins = video.event_bins.astype(np.int64, copy=False)
    support = (event_bins >= first_bin) & (event_bins <= last_bin)
    candidate_ids = np.unique(target_ids[labels & support & (target_ids > 0)])
    candidates = []
    for target_id in candidate_ids.tolist():
        target_mask = labels & support & (target_ids == int(target_id))
        bins = np.unique(event_bins[target_mask])
        if bins.size < min_track_bins:
            continue
        centers = {}
        for bin_index in bins.tolist():
            mask = target_mask & (event_bins == int(bin_index))
            centers[int(bin_index)] = video.locations[mask, :2].mean(axis=0)
        ordered = sorted(centers)
        adjacent = [
            (previous, current)
            for previous, current in zip(ordered, ordered[1:])
            if current == previous + 1
        ]
        if not adjacent:
            continue
        # Do not retain the full event-sized mask for every candidate.  Dense
        # videos contain hundreds of thousands of events; rebuilding the mask
        # only for the selected target keeps the augmentation bounded.
        candidates.append((int(target_id), centers))
    if not candidates:
        return None

    target_id, centers = candidates[
        int(rng.integers(len(candidates)))
    ]
    target_mask = labels & support & (target_ids == int(target_id))
    ordered = sorted(centers)
    direction = centers[ordered[-1]] - centers[ordered[0]]
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-6:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
        direction_norm = 1.0
    residual = direction / direction_norm * residual_speed
    pivot = float(sequence_start + (sequence_length - 1) * 0.5)
    event_bins_for_target = event_bins[target_mask].astype(np.float64)
    offsets = np.rint((event_bins_for_target - pivot)[:, None] * residual)
    original = video.locations[target_mask, :2].astype(np.int64, copy=False)
    shifted = original + offsets.astype(np.int64)
    if (
        (shifted[:, 0] < 0).any()
        or (shifted[:, 0] >= width).any()
        or (shifted[:, 1] < 0).any()
        or (shifted[:, 1] >= height).any()
    ):
        return None
    if np.array_equal(shifted, original):
        return None

    locations = video.locations.copy()
    locations[target_mask, :2] = shifted
    whole_t = max(
        int(len(video.event_indices_by_bin) * temporal_bin_size),
        int(video.locations[:, 2].max()) + 1 if video.locations.size else 1,
    )
    return temporal_frame_video_from_events(
        name=video.name,
        locations=locations,
        polarities=video.polarities,
        temporal_bin_size=temporal_bin_size,
        whole_t=whole_t,
        labels=video.labels,
        target_ids=video.target_ids,
    )
