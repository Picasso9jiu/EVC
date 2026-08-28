"""Evaluate a checkpoint on the Challenge 2 validation split.

Unlike ``test.py``, this script reads ``val/`` rather than ``test/`` and
does not write prediction text files. It uses the configured prediction
threshold and the same IoU, Acc, Pd, and Fa definitions as the Challenge 2
submission script and the project's evaluator.
"""

import torch
import tqdm

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet
from utils.challenge_eval import add_batch_to_evaluator, evaluate_challenge_metrics
from utils.component_background_verifier import ComponentBackgroundVerifier
from utils.density_threshold import DensityAdaptiveThresholdConfig
from utils.ensemble import ChallengePredictor
from utils.eval import evalute
from utils.inference_chunks import (
    InferenceChunkConfig,
    evaluation_batch_from_sample,
)
from utils.postprocess import ChallengePostprocessor
from utils.track_quality_bonus import P32TrackQualityBonus
from utils.spatial_tta import HorizontalFlipTTAConfig
from utils.temporal_frame_inference import (
    TemporalFrameInferenceConfig,
    blend_temporal_frame_scores,
    load_temporal_frame_model,
    predict_temporal_frame_scores,
    temporal_frame_video_from_sample,
)
from utils.temporal_memory_inference import (
    TemporalMemoryInferenceConfig,
    TemporalPhaseTTAConfig,
    TemporalReverseTTAConfig,
    TemporalMultiPhaseTTAConfig,
    SpatialPhaseTTAConfig,
    blend_temporal_memory_scores,
    load_temporal_memory_model,
    predict_temporal_memory_scores,
    predict_temporal_memory_scores_with_horizontal_flip,
    predict_temporal_memory_scores_with_temporal_phase,
    predict_temporal_memory_scores_with_temporal_multiphase,
    spatial_phase_shift_temporal_memory_video,
    temporal_phase_shift_temporal_memory_video,
    temporal_reverse_temporal_memory_video,
)
from utils.tta_inference import predict_sample_scores


PREDICTION_THRESHOLD = float(cfg.prediction_threshold)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Challenge 2 inference.")
    if not cfg.eval or not cfg.roc:
        raise ValueError("Set TEST.eval: True and TEST.roc: True in the config.")

    device = torch.device("cuda:0")
    threshold_policy = DensityAdaptiveThresholdConfig.from_cfg(cfg)
    chunk_config = InferenceChunkConfig.from_cfg(cfg)
    tta_config = HorizontalFlipTTAConfig.from_cfg(cfg)
    phase_tta_config = TemporalPhaseTTAConfig.from_cfg(cfg)
    reverse_tta_config = TemporalReverseTTAConfig.from_cfg(cfg)
    multiphase_tta_config = TemporalMultiPhaseTTAConfig.from_cfg(cfg)
    spatial_phase_tta_config = SpatialPhaseTTAConfig.from_cfg(cfg)
    temporal_frame_config = TemporalFrameInferenceConfig.from_cfg(cfg)
    temporal_memory_config = TemporalMemoryInferenceConfig.from_cfg(cfg)
    if temporal_frame_config.enabled and temporal_memory_config.enabled:
        raise ValueError(
            'TEMPORAL_FRAME and TEMPORAL_MEMORY cannot be enabled together.'
        )
    if tta_config.enabled and (phase_tta_config.enabled or multiphase_tta_config.enabled):
        raise ValueError('P14 horizontal flip and temporal phase TTA are mutually exclusive.')
    if phase_tta_config.enabled and multiphase_tta_config.enabled:
        raise ValueError('P41 and P50 temporal phase TTA are mutually exclusive.')
    if reverse_tta_config.enabled and spatial_phase_tta_config.enabled:
        raise ValueError('P54 temporal reversal and P44 spatial phase TTA are mutually exclusive.')
    fine_detail_bin_ratio = 1
    if temporal_frame_config.fine_detail_enabled:
        if (
            temporal_frame_config.fine_temporal_bin_size
            > int(cfg.temporal_frame_bin_size)
            or int(cfg.temporal_frame_bin_size)
            % temporal_frame_config.fine_temporal_bin_size != 0
        ):
            raise ValueError(
                'TEMPORAL_FRAME.fine_temporal_bin_size must be a positive '
                'divisor no greater than temporal_frame_bin_size.'
            )
        fine_detail_bin_ratio = (
            int(cfg.temporal_frame_bin_size)
            // temporal_frame_config.fine_temporal_bin_size
        )
    temporal_frame_only = temporal_frame_config.frame_only
    temporal_memory_only = temporal_memory_config.memory_only
    full_stream_only = temporal_frame_only or temporal_memory_only
    predictor = None
    if not full_stream_only:
        predictor = ChallengePredictor(cfg, device, evspsegnet)
    temporal_frame_model = None
    if temporal_frame_config.enabled:
        temporal_frame_model, _ = load_temporal_frame_model(
            temporal_frame_config.model_path,
            device,
            cfg.temporal_frame_context_bins,
            cfg.temporal_frame_width,
            temporal_frame_config.local_contrast_enabled,
            temporal_frame_config.local_contrast_kernel_size,
            temporal_frame_config.motion_persistence_enabled,
            temporal_frame_config.motion_persistence_radius_per_bin,
            temporal_frame_config.fine_detail_enabled,
            temporal_frame_config.fine_temporal_bin_size,
            temporal_frame_config.fine_context_bins,
            temporal_frame_config.target_center_enabled,
            temporal_frame_config.confidence_head_enabled,
            temporal_frame_config.density_calibration_enabled,
        )
    temporal_memory_model = None
    temporal_memory_secondary_model = None
    temporal_memory_blend_model = None
    temporal_memory_dense_specialist_model = None
    temporal_memory_fine_time_expert_model = None
    temporal_memory_phase_specialist_model = None
    if temporal_memory_config.enabled:
        if int(cfg.temporal_memory_context_bins) % 2 == 0:
            raise ValueError('TEMPORAL_MEMORY.context_bins must be odd.')
        if int(cfg.temporal_memory_sequence_length) <= 1:
            raise ValueError(
                'TEMPORAL_MEMORY.sequence_length must exceed one.'
            )
        temporal_memory_model, _ = load_temporal_memory_model(
            temporal_memory_config.model_path,
            device,
            cfg.temporal_memory_context_bins,
            cfg.temporal_memory_width,
            cfg.temporal_memory_sequence_length,
        )
        if temporal_memory_config.has_secondary_model:
            temporal_memory_secondary_model, _ = load_temporal_memory_model(
                temporal_memory_config.secondary_model_path,
                device,
                cfg.temporal_memory_context_bins,
                None,
                None,
            )
        if temporal_memory_config.has_blend_model:
            temporal_memory_blend_model, _ = load_temporal_memory_model(
                temporal_memory_config.blend_model_path,
                device,
                cfg.temporal_memory_context_bins,
                None,
                None,
            )
        if temporal_memory_config.dense_specialist_enabled:
            temporal_memory_dense_specialist_model, _ = load_temporal_memory_model(
                temporal_memory_config.dense_specialist_model_path,
                device,
                cfg.temporal_memory_context_bins,
                None,
                None,
            )
        if temporal_memory_config.fine_time_expert_enabled:
            temporal_memory_fine_time_expert_model, _ = load_temporal_memory_model(
                temporal_memory_config.fine_time_expert_model_path,
                device,
                temporal_memory_config.fine_time_expert_context_bins,
                cfg.temporal_memory_width,
                temporal_memory_config.fine_time_expert_sequence_length,
            )
        if temporal_memory_config.phase_specialist_enabled:
            temporal_memory_phase_specialist_model, _ = load_temporal_memory_model(
                temporal_memory_config.phase_specialist_model_path,
                device,
                cfg.temporal_memory_context_bins,
                None,
                None,
            )
            if not phase_tta_config.enabled:
                raise ValueError('M49 phase specialist requires P41 temporal phase.')
            if (
                temporal_memory_config.phase_specialist_offset
                != phase_tta_config.phase_offset
            ):
                raise ValueError('M49 specialist and P41 must use the same phase offset.')
            if (
                temporal_memory_config.phase_specialist_event_count_cutoff
                != phase_tta_config.min_event_count
            ):
                raise ValueError('M49 specialist and P41 must use the same density route.')
            if abs(
                temporal_memory_config.phase_specialist_weight
                - (1.0 - phase_tta_config.original_weight)
            ) > 1e-8:
                raise ValueError('M49 specialist weight must replace the P41 phase weight.')
            if (
                spatial_phase_tta_config.enabled
                or temporal_memory_config.dense_specialist_enabled
                or temporal_memory_config.fine_time_expert_enabled
                or (
                    temporal_memory_config.has_blend_model
                    and not temporal_memory_config.phase_specialist_blend_compatible
                )
            ):
                raise ValueError(
                    'M49 phase specialist must not be combined with other '
                    'high-density expert or spatial-phase routes. Set '
                    'TEMPORAL_MEMORY.temporal_memory_phase_specialist_blend_compatible=true '
                    'only for a fixed post-M49 score blend.'
                )
    if threshold_policy.enabled and cfg.batch_size != 1:
        raise ValueError("P6 density-adaptive threshold requires batch_size=1.")
    if not full_stream_only and chunk_config.enabled and cfg.batch_size != 1:
        raise ValueError("P8 random chunk inference requires batch_size=1.")
    if (
        not full_stream_only
        and chunk_config.enabled
        and getattr(cfg, "p3_lite_enabled", False)
    ):
        raise ValueError("P8 random chunk inference does not support P3-Lite event frames.")
    if not full_stream_only and tta_config.enabled and cfg.batch_size != 1:
        raise ValueError("P14 horizontal-flip TTA requires batch_size=1.")
    if predictor is not None and predictor.dense_expert_config.enabled and cfg.batch_size != 1:
        raise ValueError("P20 dense-expert inference requires batch_size=1.")
    if temporal_frame_config.enabled and cfg.batch_size != 1:
        raise ValueError(
            "The temporal-frame expert requires batch_size=1."
        )
    if temporal_memory_config.enabled and cfg.batch_size != 1:
        raise ValueError(
            "The temporal-memory expert requires batch_size=1."
        )

    def predict_memory_view(model, frame_video, event_count):
        if (
            multiphase_tta_config.enabled
            and event_count > multiphase_tta_config.min_event_count
        ):
            return predict_temporal_memory_scores_with_temporal_multiphase(
                model,
                frame_video,
                device,
                cfg.temporal_memory_context_bins,
                cfg.res[0],
                cfg.res[1],
                cfg.temporal_memory_inference_batch_size,
                cfg.temporal_memory_bin_size,
                multiphase_tta_config.phase_offsets,
                cfg.temporal_memory_log_count_clip,
                temporal_multiphase_enabled=True,
                original_weight=multiphase_tta_config.original_weight,
            )
        if (
            phase_tta_config.enabled
            and event_count > phase_tta_config.min_event_count
        ):
            return predict_temporal_memory_scores_with_temporal_phase(
                model,
                frame_video,
                device,
                cfg.temporal_memory_context_bins,
                cfg.res[0],
                cfg.res[1],
                cfg.temporal_memory_inference_batch_size,
                cfg.temporal_memory_bin_size,
                phase_tta_config.phase_offset,
                cfg.temporal_memory_log_count_clip,
                temporal_phase_enabled=True,
                original_weight=phase_tta_config.original_weight,
                average_mode=phase_tta_config.average_mode,
                boundary_adaptive=phase_tta_config.boundary_adaptive,
            )
        return predict_temporal_memory_scores_with_horizontal_flip(
            model,
            frame_video,
            device,
            cfg.temporal_memory_context_bins,
            cfg.res[0],
            cfg.res[1],
            cfg.temporal_memory_inference_batch_size,
            cfg.temporal_memory_log_count_clip,
            horizontal_flip_enabled=tta_config.enabled,
            original_weight=tta_config.original_weight,
        )

    def predict_memory_scores(model, frame_video, event_count):
        original_scores = predict_memory_view(model, frame_video, event_count)
        if (
            reverse_tta_config.enabled
            and event_count > reverse_tta_config.min_event_count
        ):
            reverse_scores = predict_memory_view(
                model,
                temporal_reverse_temporal_memory_video(frame_video, cfg.whole_t),
                event_count,
            )
            original_scores = (
                original_scores * reverse_tta_config.original_weight
                + reverse_scores * (1.0 - reverse_tta_config.original_weight)
            )
        if (
            not spatial_phase_tta_config.enabled
            or event_count <= spatial_phase_tta_config.min_event_count
        ):
            return original_scores
        offset = spatial_phase_tta_config.offset
        shifted_scores = original_scores
        for x_offset, y_offset in ((offset, 0), (0, offset), (offset, offset)):
            shifted_scores = shifted_scores + predict_memory_view(
                model,
                spatial_phase_shift_temporal_memory_video(
                    frame_video,
                    cfg.res[0],
                    cfg.res[1],
                    x_offset,
                    y_offset,
                ),
                event_count,
            )
        return shifted_scores * 0.25

    def predict_fine_time_expert_scores(model, frame_video):
        return predict_temporal_memory_scores(
            model,
            frame_video,
            device,
            temporal_memory_config.fine_time_expert_context_bins,
            cfg.res[0],
            cfg.res[1],
            cfg.temporal_memory_inference_batch_size,
            cfg.temporal_memory_log_count_clip,
        )

    def predict_phase_specialist_scores(model, frame_video):
        shifted_video = temporal_phase_shift_temporal_memory_video(
            frame_video,
            cfg.temporal_memory_bin_size,
            temporal_memory_config.phase_specialist_offset,
        )
        return predict_temporal_memory_scores(
            model,
            shifted_video,
            device,
            cfg.temporal_memory_context_bins,
            cfg.res[0],
            cfg.res[1],
            cfg.temporal_memory_inference_batch_size,
            cfg.temporal_memory_log_count_clip,
        )
    if (
        not full_stream_only
        and tta_config.enabled
        and getattr(cfg, "p3_lite_enabled", False)
    ):
        raise ValueError("P14 horizontal-flip TTA does not support P3-Lite event frames.")
    if predictor is None:
        print("dict load: skipped (full-stream-only inference)")
        print("model ensemble: skipped (full-stream-only inference)")
    else:
        print("dict load:", predictor.primary_model_path)
        print("model ensemble:", predictor.describe())

    dataset = EvUAV(cfg, mode="val")
    dataset.file_list = sorted(dataset.file_list)
    if not dataset.file_list:
        raise RuntimeError(f"No validation files found in: {dataset.root}")
    print("validation root:", dataset.root)
    print("validation videos:", len(dataset.file_list))
    print("prediction threshold:", PREDICTION_THRESHOLD)
    print("threshold policy:", threshold_policy.describe(PREDICTION_THRESHOLD))
    if full_stream_only:
        print("P8 random chunk inference: skipped (full-stream-only inference)")
        if temporal_memory_config.enabled:
            print("P14 horizontal-flip TTA:", tta_config.describe())
            print("P41 temporal-phase TTA:", phase_tta_config.describe())
            print("P54 temporal-reverse TTA:", reverse_tta_config.describe())
            print("P50 temporal-multiphase TTA:", multiphase_tta_config.describe())
            print("P44 spatial-phase TTA:", spatial_phase_tta_config.describe())
        else:
            print("P14 horizontal-flip TTA: skipped (full-stream-only inference)")
            print("P41 temporal-phase TTA: skipped (full-stream-only inference)")
            print("P54 temporal-reverse TTA: skipped (full-stream-only inference)")
            print("P50 temporal-multiphase TTA: skipped (full-stream-only inference)")
            print("P44 spatial-phase TTA: skipped (full-stream-only inference)")
    else:
        print("P8 random chunk inference:", chunk_config.describe())
        print("P14 horizontal-flip TTA:", tta_config.describe())
        print("P41 temporal-phase TTA:", phase_tta_config.describe())
        print("P54 temporal-reverse TTA:", reverse_tta_config.describe())
        print("P50 temporal-multiphase TTA:", multiphase_tta_config.describe())
        print("P44 spatial-phase TTA:", spatial_phase_tta_config.describe())
    print("temporal-frame expert:", temporal_frame_config.describe())
    print("temporal-memory expert:", temporal_memory_config.describe())
    postprocessor = ChallengePostprocessor.from_cfg(cfg, PREDICTION_THRESHOLD)
    background_verifier = ComponentBackgroundVerifier.from_cfg(cfg)
    if background_verifier is not None:
        print(
            "M124 background verifier: enabled (threshold {:.2f})".format(
                background_verifier.threshold
            )
        )
    else:
        print("M124 background verifier: disabled")
    postprocess_stats = postprocessor.new_stats()
    p32_track_bonus = P32TrackQualityBonus.from_cfg(
        cfg,
        PREDICTION_THRESHOLD,
    )
    p32_track_bonus_stats = p32_track_bonus.new_stats()
    threshold_usage = {}
    print("postprocessor:", postprocessor.describe())
    print("P32 track-quality bonus:", p32_track_bonus.describe())
    evaluator = evalute(cfg)
    sample_number = 0
    p8_partitioned_videos = 0
    p8_chunk_count = 0
    dataloader = None
    sample_level_inference = (
        chunk_config.enabled
        or tta_config.enabled
        or temporal_frame_config.enabled
        or temporal_memory_config.enabled
    )
    if background_verifier is not None and not sample_level_inference:
        raise ValueError(
            "M124 background verifier requires per-video inference; enable a "
            "full-stream or sample-level route."
        )
    if not sample_level_inference:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.custom_collate,
            shuffle=False,
        )
    pbar = tqdm.tqdm(
        total=len(dataset) if sample_level_inference else len(dataloader),
        desc="video",
        unit="video",
        unit_scale=True,
        position=0,
        leave=True,
    )

    if sample_level_inference:
        for video_index in range(len(dataset)):
            sample = dataset[video_index]
            event_count = len(sample["ev_loc"])
            batch = evaluation_batch_from_sample(sample)
            frame_video = None
            fine_detail_video = None
            fine_time_memory_video = None
            if temporal_frame_config.enabled or temporal_memory_config.enabled:
                frame_video = temporal_frame_video_from_sample(
                    sample,
                    (
                        cfg.temporal_memory_bin_size
                        if temporal_memory_config.enabled
                        else cfg.temporal_frame_bin_size
                    ),
                    cfg.whole_t,
                )
                if temporal_frame_config.fine_detail_enabled:
                    fine_detail_video = temporal_frame_video_from_sample(
                        sample,
                        temporal_frame_config.fine_temporal_bin_size,
                        cfg.whole_t,
                    )
            if temporal_memory_config.fine_time_expert_enabled:
                fine_time_memory_video = temporal_frame_video_from_sample(
                    sample,
                    temporal_memory_config.fine_time_expert_bin_size,
                    cfg.whole_t,
                )
            if temporal_memory_only:
                use_phase_specialist = (
                    temporal_memory_phase_specialist_model is not None
                    and temporal_memory_config.use_phase_specialist_for_event_count(
                        event_count
                    )
                )
                if use_phase_specialist:
                    # Replace P41's untrained shifted view, rather than adding
                    # a third score stream after its calibrated probability mix.
                    predictions = predict_temporal_memory_scores(
                        temporal_memory_model,
                        frame_video,
                        device,
                        cfg.temporal_memory_context_bins,
                        cfg.res[0],
                        cfg.res[1],
                        cfg.temporal_memory_inference_batch_size,
                        cfg.temporal_memory_log_count_clip,
                    )
                    phase_scores = predict_phase_specialist_scores(
                        temporal_memory_phase_specialist_model,
                        frame_video,
                    )
                    predictions = blend_temporal_memory_scores(
                        predictions,
                        phase_scores,
                        1.0 - temporal_memory_config.phase_specialist_weight,
                    )
                else:
                    predictions = predict_memory_scores(
                        temporal_memory_model, frame_video, event_count
                    )
                use_secondary = temporal_memory_config.use_secondary_for_event_count(
                    event_count
                )
                if temporal_memory_secondary_model is not None:
                    if use_secondary:
                        predictions = predict_memory_scores(
                            temporal_memory_secondary_model,
                            frame_video,
                            event_count,
                        )
                    elif not temporal_memory_config.routes_secondary_by_event_count:
                        secondary_scores = predict_memory_scores(
                            temporal_memory_secondary_model,
                            frame_video,
                            event_count,
                        )
                        predictions = blend_temporal_memory_scores(
                            predictions,
                            secondary_scores,
                            temporal_memory_config.primary_weight,
                        )
                if temporal_memory_blend_model is not None and not use_secondary:
                    blend_scores = predict_memory_scores(
                        temporal_memory_blend_model,
                        frame_video,
                        event_count,
                    )
                    predictions = blend_temporal_memory_scores(
                        predictions,
                        blend_scores,
                        temporal_memory_config.blend_primary_weight,
                    )
                if (
                    temporal_memory_dense_specialist_model is not None
                    and temporal_memory_config.use_dense_specialist_for_event_count(
                        event_count
                    )
                ):
                    dense_specialist_scores = predict_memory_scores(
                        temporal_memory_dense_specialist_model,
                        frame_video,
                        event_count,
                    )
                    predictions = blend_temporal_memory_scores(
                        predictions,
                        dense_specialist_scores,
                        1.0 - temporal_memory_config.dense_specialist_weight,
                    )
                if (
                    temporal_memory_fine_time_expert_model is not None
                    and temporal_memory_config.use_fine_time_expert_for_event_count(
                        event_count
                    )
                ):
                    fine_time_scores = predict_fine_time_expert_scores(
                        temporal_memory_fine_time_expert_model,
                        fine_time_memory_video,
                    )
                    predictions = blend_temporal_memory_scores(
                        predictions,
                        fine_time_scores,
                        1.0 - temporal_memory_config.fine_time_expert_weight,
                    )
                chunk_count = 0
            elif temporal_frame_only:
                predictions = predict_temporal_frame_scores(
                    temporal_frame_model,
                    frame_video,
                    device,
                    cfg.temporal_frame_context_bins,
                    cfg.res[0],
                    cfg.res[1],
                    cfg.temporal_frame_inference_batch_size,
                    cfg.temporal_frame_log_count_clip,
                    temporal_frame_config.local_contrast_enabled,
                    temporal_frame_config.local_contrast_kernel_size,
                    temporal_frame_config.motion_persistence_enabled,
                    temporal_frame_config.motion_persistence_radius_per_bin,
                    temporal_frame_config.fine_detail_enabled,
                    fine_detail_video,
                    temporal_frame_config.fine_context_bins,
                    fine_detail_bin_ratio,
                )
                chunk_count = 0
            else:
                predictions, chunk_count = predict_sample_scores(
                    predictor,
                    dataset,
                    sample,
                    device,
                    chunk_config,
                    tta_config,
                )
                if temporal_frame_config.enabled:
                    frame_scores = predict_temporal_frame_scores(
                        temporal_frame_model,
                        frame_video,
                        device,
                        cfg.temporal_frame_context_bins,
                        cfg.res[0],
                        cfg.res[1],
                        cfg.temporal_frame_inference_batch_size,
                        cfg.temporal_frame_log_count_clip,
                        temporal_frame_config.local_contrast_enabled,
                        temporal_frame_config.local_contrast_kernel_size,
                        temporal_frame_config.motion_persistence_enabled,
                        temporal_frame_config.motion_persistence_radius_per_bin,
                        temporal_frame_config.fine_detail_enabled,
                        fine_detail_video,
                        temporal_frame_config.fine_context_bins,
                        fine_detail_bin_ratio,
                    )
                    predictions = blend_temporal_frame_scores(
                        predictions,
                        frame_scores,
                        temporal_frame_config.sparse_weight,
                    )
                if temporal_memory_config.enabled:
                    memory_scores = predict_memory_scores(
                        temporal_memory_model, frame_video, event_count
                    )
                    use_secondary = temporal_memory_config.use_secondary_for_event_count(
                        event_count
                    )
                    if temporal_memory_secondary_model is not None:
                        if use_secondary:
                            memory_scores = predict_memory_scores(
                                temporal_memory_secondary_model,
                                frame_video,
                                event_count,
                            )
                        elif not temporal_memory_config.routes_secondary_by_event_count:
                            secondary_scores = predict_memory_scores(
                                temporal_memory_secondary_model,
                                frame_video,
                                event_count,
                            )
                            memory_scores = blend_temporal_memory_scores(
                                memory_scores,
                                secondary_scores,
                                temporal_memory_config.primary_weight,
                            )
                    if temporal_memory_blend_model is not None and not use_secondary:
                        blend_scores = predict_memory_scores(
                            temporal_memory_blend_model,
                            frame_video,
                            event_count,
                        )
                        memory_scores = blend_temporal_memory_scores(
                            memory_scores,
                            blend_scores,
                            temporal_memory_config.blend_primary_weight,
                        )
                    if (
                        temporal_memory_dense_specialist_model is not None
                        and temporal_memory_config.use_dense_specialist_for_event_count(
                            event_count
                        )
                    ):
                        dense_specialist_scores = predict_memory_scores(
                            temporal_memory_dense_specialist_model,
                            frame_video,
                            event_count,
                        )
                        memory_scores = blend_temporal_memory_scores(
                            memory_scores,
                            dense_specialist_scores,
                            1.0 - temporal_memory_config.dense_specialist_weight,
                        )
                    if (
                        temporal_memory_fine_time_expert_model is not None
                        and temporal_memory_config.use_fine_time_expert_for_event_count(
                            event_count
                        )
                    ):
                        fine_time_scores = predict_fine_time_expert_scores(
                            temporal_memory_fine_time_expert_model,
                            fine_time_memory_video,
                        )
                        memory_scores = blend_temporal_memory_scores(
                            memory_scores,
                            fine_time_scores,
                            1.0 - temporal_memory_config.fine_time_expert_weight,
                        )
                    predictions = blend_temporal_frame_scores(
                        predictions,
                        memory_scores,
                        temporal_memory_config.sparse_weight,
                    )
            if not full_stream_only and chunk_config.should_partition(event_count):
                p8_partitioned_videos += 1
                p8_chunk_count += chunk_count
            batch_threshold = threshold_policy.threshold_for_sample(
                event_count,
                sample["evs_norm"],
                PREDICTION_THRESHOLD,
            )
            batch_postprocessor = (
                ChallengePostprocessor.from_cfg(cfg, batch_threshold)
                if threshold_policy.enabled else postprocessor
            )
            raw_predictions = predictions.detach().clone()
            predictions, batch_postprocess_stats = batch_postprocessor.apply(
                predictions,
                batch["locs"],
            )
            postprocess_stats.merge(batch_postprocess_stats)
            batch_p32_track_bonus = (
                P32TrackQualityBonus.from_cfg(cfg, batch_threshold)
                if threshold_policy.enabled else p32_track_bonus
            )
            predictions, batch_p32_track_bonus_stats = batch_p32_track_bonus.apply(
                predictions,
                batch["locs"],
            )
            p32_track_bonus_stats.merge(batch_p32_track_bonus_stats)
            if background_verifier is not None:
                verifier_result = background_verifier.apply(
                    raw_predictions,
                    predictions,
                    batch["locs"],
                    batch_threshold,
                    event_count,
                )
                predictions = verifier_result.scores
            if threshold_policy.enabled:
                predictions = (predictions >= batch_threshold).to(predictions.dtype)
            threshold_usage[batch_threshold] = threshold_usage.get(batch_threshold, 0) + 1
            sample_number = add_batch_to_evaluator(
                evaluator,
                batch,
                predictions,
                sample_number,
                batch_threshold,
            )
            pbar.update(1)
    else:
        for batch in dataloader:
            with torch.no_grad():
                p2v_map = batch["p2v_map"].long().to(device)
                predictions = predictor.predict_event_scores(
                    batch["voxel_ev"],
                    p2v_map,
                    event_frame=batch.get("event_frame"),
                    source_event_count=batch["locs"].shape[0],
                )
                batch_threshold = threshold_policy.threshold_for_sample(
                    predictions.numel(),
                    batch.get("event_features"),
                    PREDICTION_THRESHOLD,
                )
                batch_postprocessor = (
                    ChallengePostprocessor.from_cfg(cfg, batch_threshold)
                    if threshold_policy.enabled else postprocessor
                )
                predictions, batch_postprocess_stats = batch_postprocessor.apply(
                    predictions,
                    batch["locs"],
                )
                postprocess_stats.merge(batch_postprocess_stats)
                batch_p32_track_bonus = (
                    P32TrackQualityBonus.from_cfg(cfg, batch_threshold)
                    if threshold_policy.enabled else p32_track_bonus
                )
                predictions, batch_p32_track_bonus_stats = batch_p32_track_bonus.apply(
                    predictions,
                    batch["locs"],
                )
                p32_track_bonus_stats.merge(batch_p32_track_bonus_stats)
                if threshold_policy.enabled:
                    # Semantic metrics are computed after the loop with one scalar
                    # threshold, so persist the selected per-video decision here.
                    predictions = (predictions >= batch_threshold).to(predictions.dtype)
                threshold_usage[batch_threshold] = threshold_usage.get(batch_threshold, 0) + 1
                sample_number = add_batch_to_evaluator(
                    evaluator,
                    batch,
                    predictions,
                    sample_number,
                    batch_threshold,
                )
            pbar.update(1)

    pbar.close()
    print("postprocess result:", postprocess_stats.summary())
    print("P32 track-quality bonus result:", p32_track_bonus_stats.summary())
    if chunk_config.enabled and not full_stream_only:
        print(
            "P8 random chunk result: {} high-density videos, {} chunk forwards".format(
                p8_partitioned_videos,
                p8_chunk_count,
            )
        )
    if threshold_policy.enabled:
        print(
            "P6 threshold usage:",
            ", ".join(
                "{:.3f}: {} videos".format(threshold, count)
                for threshold, count in sorted(threshold_usage.items())
            ),
        )

    metrics = evaluate_challenge_metrics(evaluator, PREDICTION_THRESHOLD)

    print("\nChallenge 2 validation metrics")
    print(f"IoU:      {metrics.iou:.10f}")
    print(f"Acc:      {metrics.acc:.10f}")
    print(f"Pd:       {metrics.pd:.10f}")
    print(f"Fa:       {metrics.fa:.10e}")
    print(f"Score_Fa: {metrics.score_fa:.10f}")
    print(f"Score:    {metrics.score:.10f}")
