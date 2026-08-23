"""Label-free final-component background verifier used by M124.

The verifier is intentionally placed after the normal postprocessor.  It
only observes the current video's coordinates, timestamps and raw scores;
labels, target ids and video names are never needed at inference time.
Feature extraction mirrors the frozen M115/M124 diagnostic implementation so
that the serialized estimator can be replayed by both validation and submit
scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch


TIME_BIN_SIZE = 50
WIDTH = 346
HEIGHT = 260
BIN_COUNT = 160

M115_FEATURE_NAMES = (
    "route_m26",
    "log_video_events",
    "time_fraction",
    "centroid_x",
    "centroid_y",
    "log_component_events",
    "log_component_pixels",
    "bbox_width",
    "bbox_height",
    "bbox_fill",
    "score_min",
    "score_p10",
    "score_mean",
    "score_p90",
    "score_max",
    "score_std",
    "fraction_raw_above_threshold",
    "fraction_raw_above_095",
    "local_r2_same",
    "local_r2_prev",
    "local_r2_next",
    "local_r5_same",
    "local_r5_prev",
    "local_r5_next",
    "local_r10_same",
    "local_r10_prev",
    "local_r10_next",
    "nearest_prev_distance",
    "nearest_next_distance",
    "nearest_prev_score_delta",
    "nearest_next_score_delta",
    "near_prev_r4",
    "near_next_r4",
    "near_prev_r8",
    "near_next_r8",
)

LONG_FEATURE_NAMES = (
    "long_pixel_log_count_mean",
    "long_pixel_log_count_p90",
    "long_pixel_active_fraction_mean",
    "long_pixel_active_fraction_p90",
    "long_pixel_outside_local_fraction_mean",
    "long_pixel_outside_local_fraction_p90",
    "long_neighborhood_log_count_mean",
    "long_neighborhood_active_fraction_mean",
    "long_component_repeat_rank",
    "long_component_outside_rank",
)

FEATURE_NAMES = M115_FEATURE_NAMES + LONG_FEATURE_NAMES
SCHEMA = "ev-uav-m124-background-verifier-v1"


@dataclass
class Component:
    event_indices: np.ndarray
    time_bin: int
    m115_features: np.ndarray
    long_features: np.ndarray
    pure_background: bool | None = None
    video_index: int = 0


@dataclass
class VerifierResult:
    scores: torch.Tensor
    selected_components: int
    deleted_events: int
    deleted_background_components: int | None = None
    deleted_target_components: int | None = None


def _nearest(geometry, candidates):
    if not candidates:
        return 2.0, 0.0
    distances = np.asarray(
        [np.linalg.norm(item["centroid"] - geometry["centroid"]) for item in candidates],
        dtype=np.float64,
    )
    nearest = candidates[int(np.argmin(distances))]
    return min(float(distances.min()) / 20.0, 2.0), float(
        nearest["mean_score"] - geometry["mean_score"]
    )


def _local_count(locations, bin_indices, centroid, time_bin, radius, offset):
    candidates = bin_indices.get(time_bin + offset)
    if candidates is None or not candidates.size:
        return 0.0
    delta = locations[candidates, :2].astype(np.float64) - centroid
    count = int((np.square(delta).sum(axis=1) <= radius * radius).sum())
    return float(np.log1p(count))


def _extract_m115_components(
    scores: np.ndarray,
    locations: np.ndarray,
    baseline_mask: np.ndarray,
    event_count: int,
    labels: np.ndarray | None = None,
):
    """Rebuild the exact final-bin connected components used by M115."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    locations = np.asarray(locations, dtype=np.int64).reshape(-1, 4)[:, 1:4]
    prediction = np.asarray(baseline_mask, dtype=bool).reshape(-1)
    if not (scores.size == locations.shape[0] == prediction.size):
        raise ValueError("scores, locations and baseline mask must align")
    if labels is not None:
        labels = np.asarray(labels).reshape(-1) > 0.5

    bins = np.floor_divide(locations[:, 2], TIME_BIN_SIZE)
    geometries = []
    for time_bin in np.unique(bins[prediction]):
        positive_indices = np.flatnonzero(prediction & (bins == time_bin))
        xy = locations[positive_indices, :2]
        occupancy = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        occupancy[xy[:, 1], xy[:, 0]] = 1
        count, component_map, stats, _ = cv2.connectedComponentsWithStats(
            occupancy, connectivity=8, ltype=cv2.CV_32S
        )
        component_ids = component_map[xy[:, 1], xy[:, 0]]
        for component_id in range(1, count):
            members = positive_indices[component_ids == component_id]
            component_xy = locations[members, :2].astype(np.float64)
            geometries.append(
                {
                    "time_bin": int(time_bin),
                    "event_indices": members,
                    "centroid": component_xy.mean(axis=0),
                    "mean_score": float(scores[members].mean()),
                    "pixel_count": int(stats[component_id, cv2.CC_STAT_AREA]),
                    "bbox": (
                        int(stats[component_id, cv2.CC_STAT_LEFT]),
                        int(stats[component_id, cv2.CC_STAT_TOP]),
                        int(stats[component_id, cv2.CC_STAT_WIDTH]),
                        int(stats[component_id, cv2.CC_STAT_HEIGHT]),
                    ),
                }
            )

    max_bin = max(int(bins.max()), 1)
    bin_indices = {
        int(time_bin): np.flatnonzero(bins == time_bin)
        for time_bin in np.unique(bins)
    }
    geometry_by_bin = {}
    for item in geometries:
        geometry_by_bin.setdefault(item["time_bin"], []).append(item)

    output = []
    threshold = 0.718 if int(event_count) <= 30000 else 0.7226
    for geometry in geometries:
        members = geometry["event_indices"]
        xy = locations[members, :2].astype(np.float64)
        component_scores = scores[members]
        left, top, width, height = geometry["bbox"]
        prev_distance, prev_score_delta = _nearest(
            geometry, geometry_by_bin.get(geometry["time_bin"] - 1, [])
        )
        next_distance, next_score_delta = _nearest(
            geometry, geometry_by_bin.get(geometry["time_bin"] + 1, [])
        )
        local = []
        for radius in (2, 5, 10):
            for offset in (0, -1, 1):
                local.append(
                    _local_count(
                        locations,
                        bin_indices,
                        geometry["centroid"],
                        geometry["time_bin"],
                        radius,
                        offset,
                    )
                )
        features = np.asarray(
            (
                float(int(event_count) > 30000),
                float(np.log1p(int(event_count))),
                float(geometry["time_bin"] / max_bin),
                float(geometry["centroid"][0] / WIDTH),
                float(geometry["centroid"][1] / HEIGHT),
                float(np.log1p(members.size)),
                float(np.log1p(geometry["pixel_count"])),
                float(width / WIDTH),
                float(height / HEIGHT),
                float(geometry["pixel_count"] / max(width * height, 1)),
                float(component_scores.min()),
                float(np.percentile(component_scores, 10)),
                float(component_scores.mean()),
                float(np.percentile(component_scores, 90)),
                float(component_scores.max()),
                float(component_scores.std()),
                float((component_scores >= threshold).mean()),
                float((component_scores >= 0.95).mean()),
                *local,
                prev_distance,
                next_distance,
                prev_score_delta,
                next_score_delta,
                float(prev_distance * 20.0 <= 4.0),
                float(next_distance * 20.0 <= 4.0),
                float(prev_distance * 20.0 <= 8.0),
                float(next_distance * 20.0 <= 8.0),
            ),
            dtype=np.float32,
        )
        if features.shape != (len(M115_FEATURE_NAMES),):
            raise AssertionError("M115 feature count mismatch")
        pure_background = None
        if labels is not None:
            pure_background = bool(not labels[members].any())
        output.append(
            {
                "time_bin": int(geometry["time_bin"]),
                "event_indices": members,
                "m115_features": features,
                "pure_background": pure_background,
            }
        )
    return output


def _box_sum(image, kernel_size):
    radius = int(kernel_size) // 2
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="edge")
    integral = np.pad(
        np.cumsum(np.cumsum(padded, axis=0), axis=1),
        ((1, 0), (1, 0)),
        mode="constant",
    )
    return (
        integral[kernel_size:, kernel_size:]
        - integral[:-kernel_size, kernel_size:]
        - integral[kernel_size:, :-kernel_size]
        + integral[:-kernel_size, :-kernel_size]
    )


def _rank_fraction(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.zeros(values.shape, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.size:
        end = start + 1
        while end < sorted_values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = ranks[order[start:end]].mean()
        start = end
    return (ranks / float(values.size - 1)).astype(np.float32)


def _long_maps(locations):
    locations = np.asarray(locations, dtype=np.int64).reshape(-1, 4)[:, 1:4]
    x = locations[:, 0]
    y = locations[:, 1]
    bins = np.clip(locations[:, 2] // TIME_BIN_SIZE, 0, BIN_COUNT - 1)
    flat = y * WIDTH + x
    pixel_total = np.bincount(flat, minlength=WIDTH * HEIGHT).reshape(HEIGHT, WIDTH)
    pair = bins * (WIDTH * HEIGHT) + flat
    unique_pair = np.unique(pair)
    pixel_active = np.bincount(
        unique_pair % (WIDTH * HEIGHT), minlength=WIDTH * HEIGHT
    ).reshape(HEIGHT, WIDTH)
    counts_by_bin = np.zeros((BIN_COUNT, HEIGHT, WIDTH), dtype=np.uint16)
    np.add.at(counts_by_bin, (bins, y, x), 1)
    cumulative = np.concatenate(
        (
            np.zeros((1, HEIGHT, WIDTH), dtype=np.int32),
            np.cumsum(counts_by_bin, axis=0, dtype=np.int32),
        ),
        axis=0,
    )
    return {
        "locations": locations,
        "pixel_total": pixel_total,
        "pixel_active": pixel_active,
        "pixel_total_3": _box_sum(pixel_total.astype(np.float32), 3),
        "pixel_active_3": _box_sum(pixel_active.astype(np.float32), 3),
        "cumulative": cumulative,
    }


def _component_long_features(component, maps):
    locations = maps["locations"]
    indices = np.asarray(component["event_indices"], dtype=np.int64)
    pixels = np.unique(locations[indices, :2], axis=0)
    x = pixels[:, 0]
    y = pixels[:, 1]
    time_bin = int(component["time_bin"])
    low = max(0, time_bin - 2)
    high = min(BIN_COUNT, time_bin + 3)
    local_count = maps["cumulative"][high, y, x] - maps["cumulative"][low, y, x]
    total = maps["pixel_total"][y, x].astype(np.float64)
    active = maps["pixel_active"][y, x].astype(np.float64)
    outside = np.maximum(total - local_count.astype(np.float64), 0.0)
    outside_fraction = outside / np.maximum(total, 1.0)
    total_3 = maps["pixel_total_3"][y, x].astype(np.float64)
    active_3 = maps["pixel_active_3"][y, x].astype(np.float64)
    return np.asarray(
        (
            np.log1p(total).mean(),
            np.percentile(np.log1p(total), 90),
            (active / float(BIN_COUNT)).mean(),
            np.percentile(active / float(BIN_COUNT), 90),
            outside_fraction.mean(),
            np.percentile(outside_fraction, 90),
            np.log1p(total_3).mean(),
            (active_3 / float(BIN_COUNT)).mean(),
            0.0,
            0.0,
        ),
        dtype=np.float32,
    )


def extract_components(
    scores,
    locations,
    baseline_mask,
    event_count,
    labels=None,
    video_index=0,
):
    """Return M115+M124 components and their 45-dimensional features."""
    scores_np = torch.as_tensor(scores).detach().cpu().numpy().astype(np.float64, copy=False)
    locations_np = torch.as_tensor(locations).detach().cpu().numpy()
    labels_np = None if labels is None else torch.as_tensor(labels).detach().cpu().numpy()
    m115 = _extract_m115_components(
        scores_np, locations_np, baseline_mask, event_count, labels_np
    )
    maps = _long_maps(locations_np)
    if m115:
        long = np.stack([_component_long_features(item, maps) for item in m115])
        long[:, -2] = _rank_fraction(long[:, 6])
        long[:, -1] = _rank_fraction(long[:, 4])
    else:
        long = np.zeros((0, len(LONG_FEATURE_NAMES)), dtype=np.float32)
    components = []
    for item, long_features in zip(m115, long):
        components.append(
            Component(
                event_indices=item["event_indices"],
                time_bin=item["time_bin"],
                m115_features=item["m115_features"],
                long_features=long_features,
                pure_background=item["pure_background"],
                video_index=int(video_index),
            )
        )
    if components:
        features = np.concatenate(
            (
                np.stack([item.m115_features for item in components]),
                np.stack([item.long_features for item in components]),
            ),
            axis=1,
        )
    else:
        features = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    if features.shape[1] != len(FEATURE_NAMES):
        raise AssertionError("M124 feature count mismatch")
    return components, features


def extract_training_features(records, masks):
    """Build the frozen training matrix and pure-background labels."""
    all_components = []
    all_features = []
    all_labels = []
    for video_index, (record, mask) in enumerate(zip(records, masks)):
        components, features = extract_components(
            record["scores"],
            record["locs"],
            mask,
            int(record["event_count"]),
            labels=record["seg_label"],
            video_index=video_index,
        )
        all_components.extend(components)
        all_features.append(features)
        all_labels.extend(bool(item.pure_background) for item in components)
    if not all_features:
        raise RuntimeError("No final components were found.")
    return all_components, np.concatenate(all_features, axis=0), np.asarray(all_labels, dtype=np.int64)


def load_artifact(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("Unsupported M124 verifier artifact: {}".format(path))
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("M124 verifier feature schema mismatch")
    model = payload.get("model")
    if not hasattr(model, "predict_proba"):
        raise TypeError("M124 verifier artifact has no predict_proba model")
    return payload


class ComponentBackgroundVerifier:
    """Apply a frozen verifier to one postprocessed video."""

    def __init__(self, artifact):
        self.artifact = artifact
        self.model = artifact["model"]
        self.threshold = float(artifact["verifier_threshold"])

    @classmethod
    def from_cfg(cls, cfg):
        if not bool(getattr(cfg, "m124_background_verifier_enabled", False)):
            return None
        path = str(getattr(cfg, "m124_background_verifier_model_path", ""))
        if not path:
            raise ValueError(
                "M124 verifier is enabled but m124_background_verifier_model_path is empty"
            )
        artifact = load_artifact(path)
        configured = float(getattr(cfg, "m124_background_verifier_threshold", artifact["verifier_threshold"]))
        if abs(configured - float(artifact["verifier_threshold"])) > 1e-12:
            raise ValueError("Configured M124 threshold differs from artifact threshold")
        return cls(artifact)

    def apply(self, raw_scores, processed_scores, locations, threshold, event_count):
        processed = processed_scores.reshape(-1)
        baseline_mask = (processed.detach().cpu().numpy() >= float(threshold))
        components, features = extract_components(
            raw_scores,
            locations,
            baseline_mask,
            int(event_count),
        )
        if not components:
            return VerifierResult(processed_scores, 0, 0)
        probabilities = self.model.predict_proba(features)[:, 1]
        selected = probabilities >= self.threshold
        output = processed.clone()
        deleted_events = 0
        for component, take in zip(components, selected):
            if bool(take):
                indices = torch.as_tensor(component.event_indices, device=output.device)
                output[indices] = 0.0
                deleted_events += int(indices.numel())
        return VerifierResult(
            output.reshape(processed_scores.shape),
            int(selected.sum()),
            deleted_events,
        )


def build_reference_masks(records):
    """Build masks with the released M26/P41 postprocessor for model fitting."""
    from eval_m114_score_oracle import _postprocessor, _threshold

    masks = []
    for record in records:
        threshold = _threshold(record)
        processed, _ = _postprocessor(threshold).apply(
            record["scores"].detach().cpu().float().clone(), record["locs"]
        )
        masks.append((processed.reshape(-1) >= threshold).cpu().numpy())
    return masks
