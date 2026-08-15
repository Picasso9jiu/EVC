"""Train a bidirectional full-stream temporal-memory event segmentation model."""

import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import tqdm
import yaml

from configs.configs import cfg
from dataset.temporal_memory import (
    TemporalMemoryTrainDataset,
    temporal_memory_collate,
)
from model.modules.confidence_head import confidence_calibration_loss
from model.temporal_memory_net import BidirectionalTemporalMemoryNet
from utils.component_hard_negative import (
    component_hard_negative_loss,
    target_frame_activation_loss,
)
from utils.temporal_frame_loss import (
    build_target_center_heatmaps,
    frame_balanced_event_bce,
    hard_negative_score_loss,
    target_centroid_flow_loss,
    target_centroid_trajectory_flow_loss,
    target_center_heatmap_loss,
    target_level_presence_loss as target_level_presence_loss_fn,
    target_level_velocity_loss as target_level_velocity_loss_fn,
    trajectory_extrapolation_loss_memory,
)


def setup_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_run_directory(config):
    started_at = datetime.now().astimezone()
    run_name = '{}_seed{}_pid{}'.format(
        started_at.strftime('%Y%m%d-%H%M%S'),
        int(config.seed),
        os.getpid(),
    )
    run_dir = Path(config.model_save_root) / 'runs' / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / 'config.yaml').open('w', encoding='utf-8') as stream:
        yaml.safe_dump(
            config.resolved_config,
            stream,
            allow_unicode=True,
            sort_keys=False,
        )
    return run_dir, started_at


def save_checkpoint(checkpoint, path):
    temporary_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def build_scheduler(optimizer, config):
    scheduler_name = str(config.scheduler).lower()
    if scheduler_name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.epochs),
            eta_min=float(config.scheduler_min_lr),
        )
    if scheduler_name == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.scheduler_step_size),
            gamma=float(config.scheduler_gamma),
        )
    raise ValueError('Unsupported scheduler: {}'.format(config.scheduler))


def load_p23_base_weights(
    model,
    checkpoint_path,
    context_bins,
    width,
    density_calibration_enabled=False,
    confidence_head_enabled=False,
    target_center_enabled=False,
    target_level_enabled=False,
    center_memory_enabled=False,
    local_temporal_context_enabled=False,
):
    checkpoint_path = Path(str(checkpoint_path).strip())
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            'P23 initialization checkpoint not found: {}'.format(checkpoint_path)
        )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    saved_memory = checkpoint.get('temporal_memory')
    if saved_memory is not None:
        saved_context_bins = saved_memory.get('context_bins')
        saved_width = saved_memory.get('width')
        saved_sequence_length = saved_memory.get('sequence_length')
        saved_temporal_bin_size = saved_memory.get('temporal_bin_size')
        if (
            saved_context_bins is not None
            and int(saved_context_bins) != int(context_bins)
        ):
            raise ValueError(
                'M5 context_bins={} does not match {}.'.format(
                    saved_context_bins, context_bins
                )
            )
        if saved_width is not None and int(saved_width) != int(width):
            raise ValueError(
                'M5 width={} does not match {}.'.format(saved_width, width)
            )
        if (
            saved_sequence_length is not None
            and int(saved_sequence_length) != int(cfg.temporal_memory_sequence_length)
            and not bool(
                getattr(
                    cfg,
                    'temporal_memory_init_allow_sequence_length_transfer',
                    False,
                )
            )
        ):
            raise ValueError(
                'M5 sequence_length={} does not match {}.'.format(
                    saved_sequence_length, cfg.temporal_memory_sequence_length
                )
            )
        if (
            saved_temporal_bin_size is not None
            and int(saved_temporal_bin_size) != int(cfg.temporal_memory_bin_size)
            and not bool(
                getattr(
                    cfg,
                    'temporal_memory_init_allow_temporal_bin_transfer',
                    False,
                )
            )
        ):
            raise ValueError(
                'M5 temporal_bin_size={} does not match {}. Set '
                'TEMPORAL_MEMORY.temporal_memory_init_allow_temporal_bin_transfer=true '
                'only when the model architecture is unchanged.'.format(
                    saved_temporal_bin_size,
                    cfg.temporal_memory_bin_size,
                )
            )
        saved_density_calibration = bool(
            saved_memory.get('density_calibration_enabled', False)
        )
        saved_confidence_head = bool(
            saved_memory.get('confidence_head_enabled', False)
        )
        saved_target_center = bool(
            saved_memory.get('target_center_enabled', False)
        )
        saved_target_level = bool(
            saved_memory.get('target_level_enabled', False)
        )
        saved_center_memory = bool(
            saved_memory.get('center_memory_enabled', False)
        )
        saved_temporal_attention = bool(
            saved_memory.get('temporal_attention_enabled', False)
        )
        saved_attention_relative_bias = bool(
            saved_memory.get('attention_relative_bias_enabled', False)
        )
        saved_attention_relative_bias_max_distance = int(
            saved_memory.get('attention_relative_bias_max_distance', 8)
        )
        saved_advection_alignment = bool(
            saved_memory.get('advection_alignment_enabled', False)
        )
        saved_advection_max_flow = float(
            saved_memory.get('advection_max_flow', 0.0)
        )
        saved_fine_temporal_memory = bool(
            saved_memory.get('fine_temporal_memory_enabled', False)
        )
        saved_fine_advection_max_flow = float(
            saved_memory.get('fine_advection_max_flow', 0.0)
        )
        if saved_density_calibration != bool(density_calibration_enabled):
            raise ValueError(
                'M5 density calibration={} does not match configured {}.'.format(
                    saved_density_calibration, density_calibration_enabled
                )
            )
        adding_confidence_head = (
            bool(confidence_head_enabled) and not saved_confidence_head
        )
        adding_target_center = (
            bool(target_center_enabled) and not saved_target_center
        )
        adding_target_level = (
            bool(target_level_enabled) and not saved_target_level
        )
        adding_center_memory = (
            bool(center_memory_enabled) and not saved_center_memory
        )
        if (
            saved_confidence_head != bool(confidence_head_enabled)
            and not adding_confidence_head
        ):
            raise ValueError(
                'M5 confidence head={} does not match configured {}.'.format(
                    saved_confidence_head, confidence_head_enabled
                )
            )
        if (
            saved_target_center != bool(target_center_enabled)
            and not adding_target_center
        ):
            raise ValueError(
                'M5 target-centre head={} does not match configured {}.'.format(
                    saved_target_center,
                    target_center_enabled,
                )
            )
        if (
            saved_target_level != bool(target_level_enabled)
            and not adding_target_level
        ):
            raise ValueError(
                'M5 target-level head={} does not match configured {}.'.format(
                    saved_target_level,
                    target_level_enabled,
                )
            )
        if (
            saved_center_memory != bool(center_memory_enabled)
            and not adding_center_memory
        ):
            raise ValueError(
                'M5 center memory={} does not match configured {}.'.format(
                    saved_center_memory,
                    center_memory_enabled,
                )
            )
        configured_temporal_attention = bool(
            getattr(cfg, 'temporal_memory_temporal_attention_enabled', False)
        )
        configured_attention_relative_bias = bool(
            getattr(cfg, 'temporal_memory_attention_relative_bias_enabled', False)
        )
        configured_attention_relative_bias_max_distance = int(
            getattr(cfg, 'temporal_memory_attention_relative_bias_max_distance', 8)
        )
        adding_temporal_attention = (
            configured_temporal_attention and not saved_temporal_attention
        )
        configured_advection_alignment = bool(
            getattr(cfg, 'temporal_memory_advection_alignment_enabled', False)
        )
        configured_advection_max_flow = float(
            getattr(cfg, 'temporal_memory_advection_max_flow', 0.0)
        )
        adding_advection_alignment = (
            configured_advection_alignment and not saved_advection_alignment
        )
        configured_fine_temporal_memory = bool(
            getattr(cfg, 'temporal_memory_fine_temporal_memory_enabled', False)
        )
        configured_fine_advection_max_flow = float(
            getattr(cfg, 'temporal_memory_fine_advection_max_flow', 0.0)
        )
        adding_fine_temporal_memory = (
            configured_fine_temporal_memory and not saved_fine_temporal_memory
        )
        adding_local_temporal_context = bool(
            local_temporal_context_enabled
            and not bool(saved_memory.get('local_temporal_context_enabled', False))
        )
        if saved_temporal_attention and not configured_temporal_attention:
            raise ValueError(
                'M5 temporal attention={} does not match configured {}.'.format(
                    saved_temporal_attention, configured_temporal_attention
                )
            )
        if saved_temporal_attention and (
            saved_attention_relative_bias != configured_attention_relative_bias
            or (
                saved_attention_relative_bias
                and saved_attention_relative_bias_max_distance
                != configured_attention_relative_bias_max_distance
            )
        ):
            raise ValueError(
                'Temporal-attention relative-bias metadata does not match '
                'the configured architecture.'
            )
        if saved_advection_alignment and not configured_advection_alignment:
            raise ValueError(
                'M5 advection alignment={} does not match configured {}.'.format(
                    saved_advection_alignment, configured_advection_alignment
                )
            )
        if (
            saved_advection_alignment
            and saved_advection_max_flow != configured_advection_max_flow
        ):
            raise ValueError(
                'Advection max-flow metadata does not match the configured architecture.'
            )
        if saved_fine_temporal_memory and not configured_fine_temporal_memory:
            raise ValueError(
                'Fine temporal memory={} does not match configured {}.'.format(
                    saved_fine_temporal_memory,
                    configured_fine_temporal_memory,
                )
            )
        if (
            saved_fine_temporal_memory
            and saved_fine_advection_max_flow != configured_fine_advection_max_flow
        ):
            raise ValueError(
                'Fine advection max-flow metadata does not match the configured '
                'architecture.'
            )
        # The attention branch is zero-initialized at construction time, so it
        # can be attached safely to an existing ConvGRU checkpoint. Its
        # parameters are intentionally the only new missing keys allowed here.
        load_result = model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=not (
                adding_confidence_head
                or adding_target_center
                or adding_target_level
                or adding_center_memory
                or adding_temporal_attention
                or adding_advection_alignment
                or adding_fine_temporal_memory
                or adding_local_temporal_context
            ),
        )
        if (
            adding_confidence_head
            or adding_target_center
            or adding_target_level
            or adding_center_memory
            or adding_temporal_attention
            or adding_advection_alignment
            or adding_fine_temporal_memory
            or adding_local_temporal_context
        ):
            expected_missing = set()
            if adding_confidence_head:
                expected_missing.update(
                    'base.confidence_head.' + name
                    for name in model.base.confidence_head.state_dict()
                )
            if adding_target_center:
                expected_missing.update(
                    'base.target_center_head.' + name
                    for name in model.base.target_center_head.state_dict()
                )
                expected_missing.update(
                    'base.target_center_residual.' + name
                    for name in model.base.target_center_residual.state_dict()
                )
            if adding_target_level:
                for module_name in (
                    'target_level_center_head',
                    'target_level_presence_head',
                    'target_level_velocity_head',
                ):
                    module = getattr(model, module_name)
                    expected_missing.update(
                        module_name + '.' + name
                        for name in module.state_dict()
                    )
            if adding_center_memory:
                for module_name in (
                    'center_memory_projection_in',
                    'center_memory_forward',
                    'center_memory_backward',
                    'center_memory_projection_out',
                    'center_event_projection',
                ):
                    module = getattr(model, module_name)
                    expected_missing.update(
                        module_name + '.' + name
                        for name in module.state_dict()
                    )
            if adding_temporal_attention:
                expected_missing.update(
                    'temporal_attn.' + name
                    for name in model.temporal_attn.state_dict()
                )
            if adding_advection_alignment:
                expected_missing.update(
                    'flow_head.' + name
                    for name in model.flow_head.state_dict()
                )
            if adding_fine_temporal_memory:
                for module_name in (
                    'fine_forward_memory',
                    'fine_backward_memory',
                    'fine_memory_projection',
                    'fine_flow_head',
                ):
                    module = getattr(model, module_name)
                    expected_missing.update(
                        module_name + '.' + name
                        for name in module.state_dict()
                    )
            if adding_local_temporal_context:
                expected_missing.update(
                    'base.local_temporal_context_adapter.' + name
                    for name in model.base.local_temporal_context_adapter.state_dict()
                )
            if (
                set(load_result.missing_keys) != expected_missing
                or load_result.unexpected_keys
            ):
                raise RuntimeError(
                    'Only newly attached branches may be missing when '
                    'initializing from a complete temporal-memory checkpoint. '
                    'Missing={}, unexpected={}.'.format(
                        load_result.missing_keys,
                        load_result.unexpected_keys,
                    )
                )
        return checkpoint_path

    saved = checkpoint.get('temporal_frame', {})
    if saved.get('context_bins') is not None and int(
        saved['context_bins']
    ) != int(context_bins):
        raise ValueError(
            'P23 context_bins={} does not match {}.'.format(
                saved['context_bins'], context_bins
            )
        )
    if saved.get('width') is not None and int(saved['width']) != int(width):
        raise ValueError(
            'P23 width={} does not match {}.'.format(saved['width'], width)
        )
    # A pure-P23 checkpoint has no density-calibrator or confidence-head
    # keys; leave those at their safe identity/zero init instead.
    model.base.load_state_dict(
        checkpoint['model_state_dict'],
        strict=not bool(
            density_calibration_enabled
            or confidence_head_enabled
            or target_center_enabled
            or local_temporal_context_enabled
        ),
    )
    return checkpoint_path


def build_optimizer(
    model,
    config,
    confidence_only_enabled=False,
    fine_memory_only_enabled=False,
    center_memory_only_enabled=False,
    advection_flow_only_enabled=False,
    memory_only_enabled=False,
    local_temporal_context_only_enabled=False,
):
    base_multiplier = float(config.temporal_memory_base_lr_multiplier)
    memory_multiplier = float(config.temporal_memory_memory_lr_multiplier)
    attention_multiplier = float(
        getattr(config, 'temporal_memory_attention_lr_multiplier', 1.0)
    )
    advection_multiplier = float(
        getattr(config, 'temporal_memory_advection_alignment_lr_multiplier', 0.5)
    )
    fine_memory_multiplier = float(
        getattr(config, 'temporal_memory_fine_memory_lr_multiplier', 1.0)
    )
    confidence_multiplier = float(
        getattr(config, 'temporal_memory_confidence_lr_multiplier', 1.0)
    )
    center_memory_multiplier = float(
        getattr(config, 'temporal_memory_center_memory_lr_multiplier', 1.0)
    )
    target_level_multiplier = float(
        getattr(config, 'temporal_memory_target_level_lr_multiplier', 0.5)
    )
    local_temporal_context_multiplier = float(
        getattr(config, 'temporal_memory_local_temporal_context_lr_multiplier', 1.0)
    )
    if (
        base_multiplier <= 0.0
        or memory_multiplier <= 0.0
        or attention_multiplier <= 0.0
        or advection_multiplier <= 0.0
        or fine_memory_multiplier <= 0.0
        or confidence_multiplier <= 0.0
        or center_memory_multiplier <= 0.0
        or target_level_multiplier <= 0.0
        or local_temporal_context_multiplier <= 0.0
    ):
        raise ValueError('Temporal-memory learning-rate multipliers must be positive.')
    confidence_parameters = []
    if model.confidence_head_enabled:
        confidence_parameters = list(model.base.confidence_head.parameters())
    confidence_parameter_ids = {id(parameter) for parameter in confidence_parameters}
    fine_parameters = []
    if getattr(model, 'fine_temporal_memory_enabled', False):
        for module in (
            model.fine_forward_memory,
            model.fine_backward_memory,
            model.fine_memory_projection,
            model.fine_flow_head,
        ):
            fine_parameters += list(module.parameters())
    center_parameters = []
    if getattr(model, 'center_memory_enabled', False):
        for module in (
            model.base.target_center_head,
            model.base.target_center_residual,
            model.center_memory_projection_in,
            model.center_memory_forward,
            model.center_memory_backward,
            model.center_memory_projection_out,
            model.center_event_projection,
        ):
            center_parameters += list(module.parameters())
    target_level_parameters = []
    if getattr(model, 'target_level_enabled', False):
        for module in (
            model.target_level_center_head,
            model.target_level_presence_head,
            model.target_level_velocity_head,
        ):
            target_level_parameters += list(module.parameters())
    if confidence_only_enabled:
        if not confidence_parameters:
            raise ValueError(
                'Confidence-only mode requires the confidence head to be enabled.'
            )
        return optim.AdamW(
            [
                {
                    'name': 'confidence',
                    'params': confidence_parameters,
                    'lr': float(config.lr) * confidence_multiplier,
                }
            ],
            weight_decay=1e-4,
        )
    local_temporal_context_parameters = []
    if getattr(model, 'local_temporal_context_enabled', False):
        local_temporal_context_parameters = list(
            model.base.local_temporal_context_adapter.parameters()
        )
    if local_temporal_context_only_enabled:
        if not local_temporal_context_parameters:
            raise ValueError(
                'Local-context-only mode requires the local context adapter.'
            )
        return optim.AdamW(
            [
                {
                    'name': 'local_temporal_context',
                    'params': local_temporal_context_parameters,
                    'lr': float(config.lr) * local_temporal_context_multiplier,
                }
            ],
            weight_decay=1e-4,
        )
    if fine_memory_only_enabled:
        if not fine_parameters:
            raise ValueError(
                'Fine-memory-only mode requires fine temporal memory to be enabled.'
            )
        return optim.AdamW(
            [
                {
                    'name': 'fine_memory',
                    'params': fine_parameters,
                    'lr': float(config.lr) * fine_memory_multiplier,
                }
            ],
            weight_decay=1e-4,
        )
    if center_memory_only_enabled:
        if not center_parameters:
            raise ValueError(
                'Center-memory-only mode requires target-centre memory.'
            )
        return optim.AdamW(
            [
                {
                    'name': 'center_memory',
                    'params': center_parameters,
                    'lr': float(config.lr) * center_memory_multiplier,
                }
            ],
            weight_decay=1e-4,
        )
    if advection_flow_only_enabled:
        if not getattr(model, 'advection_alignment_enabled', False):
            raise ValueError(
                'Flow-only mode requires advection alignment to be enabled.'
            )
        return optim.AdamW(
            [
                {
                    'name': 'advection_flow',
                    'params': list(model.flow_head.parameters()),
                    'lr': float(config.lr) * advection_multiplier,
                }
            ],
            weight_decay=1e-4,
        )
    base_parameters = [
        parameter
        for parameter in model.base.parameters()
        if id(parameter) not in confidence_parameter_ids
    ]
    memory_parameters = list(model.forward_memory.parameters())
    memory_parameters += list(model.backward_memory.parameters())
    memory_parameters += list(model.memory_projection.parameters())
    attention_parameters = []
    if getattr(model, 'temporal_attention_enabled', False):
        attention_parameters = list(model.temporal_attn.parameters())
    if memory_only_enabled:
        parameter_groups = [
            {
                'name': 'memory',
                'params': memory_parameters,
                'lr': float(config.lr) * memory_multiplier,
            },
        ]
        if attention_parameters:
            parameter_groups.append(
                {
                    'name': 'attention',
                    'params': attention_parameters,
                    'lr': float(config.lr) * attention_multiplier,
                }
            )
        return optim.AdamW(parameter_groups, weight_decay=1e-4)
    parameter_groups = [
        {
            'name': 'base',
            'params': base_parameters,
            'lr': float(config.lr) * base_multiplier,
        },
    ]
    if confidence_parameters:
        parameter_groups.append(
            {
                'name': 'confidence',
                'params': confidence_parameters,
                'lr': float(config.lr) * confidence_multiplier,
            }
        )
    if local_temporal_context_parameters:
        parameter_groups.append(
            {
                'name': 'local_temporal_context',
                'params': local_temporal_context_parameters,
                'lr': float(config.lr) * local_temporal_context_multiplier,
            }
        )
    parameter_groups.append(
        {
            'name': 'memory',
            'params': memory_parameters,
            'lr': float(config.lr) * memory_multiplier,
        }
    )
    if attention_parameters:
        parameter_groups.append(
            {
                'name': 'attention',
                'params': attention_parameters,
                'lr': float(config.lr) * attention_multiplier,
            }
        )
    if getattr(model, 'advection_alignment_enabled', False):
        parameter_groups.append(
            {
                'name': 'advection',
                'params': list(model.flow_head.parameters()),
                'lr': float(config.lr) * advection_multiplier,
            }
        )
    if getattr(model, 'fine_temporal_memory_enabled', False):
        parameter_groups.append(
            {
                'name': 'fine_memory',
                'params': fine_parameters,
                'lr': float(config.lr) * fine_memory_multiplier,
            }
        )
    if getattr(model, 'center_memory_enabled', False):
        parameter_groups.append(
            {
                'name': 'center_memory',
                'params': center_parameters,
                'lr': float(config.lr) * center_memory_multiplier,
            }
            )
    if target_level_parameters:
        parameter_groups.append(
            {
                'name': 'target_level',
                'params': target_level_parameters,
                'lr': float(config.lr) * target_level_multiplier,
            }
        )
    return optim.AdamW(parameter_groups, weight_decay=1e-4)


def memory_config_summary(config):
    return (
        'enabled (bin_size={}, context_bins={}, width={}, sequence_length={}, '
        'views_per_video={}, positive_frame_probability={}, '
        'target_positive_loss_mass={}, max_positive_weight={}, '
        'base_lr_multiplier={}, memory_lr_multiplier={}, '
        'confidence_lr_multiplier={}, attention_lr_multiplier={}, '
        'attention_enabled={}, attention_output_init_std={}, '
        'attention_relative_bias_enabled={}, attention_relative_bias_max_distance={}, '
        'advection_enabled={}, advection_loss_weight={}, advection_lr_multiplier={}, '
        'advection_max_flow={}, target_flow_enabled={}, target_flow_weight={}, '
        'fine_memory_enabled={}, fine_advection_max_flow={}, '
        'fine_target_flow_enabled={}, fine_target_flow_weight={}, '
        'fine_memory_lr_multiplier={})'
    ).format(
        config.temporal_memory_bin_size,
        config.temporal_memory_context_bins,
        config.temporal_memory_width,
        config.temporal_memory_sequence_length,
        config.temporal_memory_train_views_per_video,
        config.temporal_memory_positive_frame_probability,
        config.temporal_memory_target_positive_loss_mass,
        config.temporal_memory_max_positive_weight,
        config.temporal_memory_base_lr_multiplier,
        config.temporal_memory_memory_lr_multiplier,
        getattr(config, 'temporal_memory_confidence_lr_multiplier', 1.0),
        getattr(config, 'temporal_memory_attention_lr_multiplier', 1.0),
        bool(getattr(config, 'temporal_memory_temporal_attention_enabled', False)),
        getattr(config, 'temporal_memory_attention_output_init_std', 0.0),
        bool(
            getattr(config, 'temporal_memory_attention_relative_bias_enabled', False)
        ),
        getattr(config, 'temporal_memory_attention_relative_bias_max_distance', 8),
        bool(getattr(config, 'temporal_memory_advection_alignment_enabled', False)),
        getattr(config, 'temporal_memory_advection_alignment_loss_weight', 0.05),
        getattr(config, 'temporal_memory_advection_alignment_lr_multiplier', 0.5),
        getattr(config, 'temporal_memory_advection_max_flow', 0.0),
        bool(getattr(config, 'temporal_memory_advection_target_flow_enabled', False)),
        getattr(config, 'temporal_memory_advection_target_flow_weight', 0.0),
        bool(getattr(config, 'temporal_memory_fine_temporal_memory_enabled', False)),
        getattr(config, 'temporal_memory_fine_advection_max_flow', 0.0),
        bool(getattr(config, 'temporal_memory_fine_target_flow_enabled', False)),
        getattr(config, 'temporal_memory_fine_target_flow_weight', 0.0),
        getattr(config, 'temporal_memory_fine_memory_lr_multiplier', 1.0),
    )


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for temporal-memory training.')
    if not bool(cfg.temporal_memory_enabled):
        raise ValueError('Set TEMPORAL_MEMORY.temporal_memory_enabled=true.')
    if int(cfg.temporal_memory_context_bins) % 2 == 0:
        raise ValueError('TEMPORAL_MEMORY.context_bins must be odd.')
    if int(cfg.temporal_memory_sequence_length) <= 1:
        raise ValueError('TEMPORAL_MEMORY.sequence_length must exceed one.')
    if int(cfg.temporal_memory_train_workers) != 0 and bool(
        cfg.temporal_memory_cache_all_videos
    ):
        raise ValueError(
            'Use TEMPORAL_MEMORY.train_workers=0 when cache_all_videos=true.'
        )
    if int(cfg.epochs) <= 0:
        raise ValueError('TRAIN.epochs must be positive.')

    setup_seed(cfg.seed)
    device = torch.device('cuda:0')
    run_dir, started_at = create_run_directory(cfg)
    dataset = TemporalMemoryTrainDataset(
        root=Path(cfg.root) / 'train',
        whole_t=cfg.whole_t,
        temporal_bin_size=cfg.temporal_memory_bin_size,
        context_bins=cfg.temporal_memory_context_bins,
        sequence_length=cfg.temporal_memory_sequence_length,
        width=cfg.res[0],
        height=cfg.res[1],
        views_per_video=cfg.temporal_memory_train_views_per_video,
        positive_frame_probability=cfg.temporal_memory_positive_frame_probability,
        random_seed=cfg.seed,
        log_count_clip=cfg.temporal_memory_log_count_clip,
        cache_all_videos=cfg.temporal_memory_cache_all_videos,
        cache_video_count=cfg.temporal_memory_cache_video_count,
        dense_sampling_enabled=getattr(
            cfg,
            'temporal_memory_dense_sampling_enabled',
            False,
        ),
        dense_event_count_cutoff=getattr(
            cfg,
            'temporal_memory_dense_event_count_cutoff',
            200000,
        ),
        dense_view_multiplier=getattr(
            cfg,
            'temporal_memory_dense_view_multiplier',
            2,
        ),
        mid_density_sampling_enabled=getattr(
            cfg,
            'temporal_memory_mid_density_sampling_enabled',
            False,
        ),
        mid_density_min_event_count=getattr(
            cfg,
            'temporal_memory_mid_density_min_event_count',
            30000,
        ),
        mid_density_max_event_count=getattr(
            cfg,
            'temporal_memory_mid_density_max_event_count',
            200000,
        ),
        mid_density_view_multiplier=getattr(
            cfg,
            'temporal_memory_mid_density_view_multiplier',
            2,
        ),
        dense_only_enabled=getattr(
            cfg,
            'temporal_memory_dense_only_enabled',
            False,
        ),
        dense_only_event_count_cutoff=getattr(
            cfg,
            'temporal_memory_dense_only_event_count_cutoff',
            100000,
        ),
        low_density_only_enabled=getattr(
            cfg,
            'temporal_memory_low_density_only_enabled',
            False,
        ),
        low_density_only_event_count_cutoff=getattr(
            cfg,
            'temporal_memory_low_density_only_event_count_cutoff',
            30000,
        ),
        max_videos_per_epoch=getattr(
            cfg,
            'temporal_memory_max_videos_per_epoch',
            0,
        ),
        motion_sampling_enabled=getattr(
            cfg,
            'temporal_memory_motion_sampling_enabled',
            False,
        ),
        motion_sampling_min_event_count=getattr(
            cfg,
            'temporal_memory_motion_sampling_min_event_count',
            30000,
        ),
        motion_sampling_min_displacement=getattr(
            cfg,
            'temporal_memory_motion_sampling_min_displacement',
            4.0,
        ),
        motion_sampling_probability=getattr(
            cfg,
            'temporal_memory_motion_sampling_probability',
            0.50,
        ),
        motion_sampling_extra_views_only=getattr(
            cfg,
            'temporal_memory_motion_sampling_extra_views_only',
            True,
        ),
        trajectory_augmentation_enabled=getattr(
            cfg,
            'temporal_memory_trajectory_augmentation_enabled',
            False,
        ),
        trajectory_augmentation_min_event_count=getattr(
            cfg,
            'temporal_memory_trajectory_augmentation_min_event_count',
            30000,
        ),
        trajectory_augmentation_probability=getattr(
            cfg,
            'temporal_memory_trajectory_augmentation_probability',
            0.50,
        ),
        trajectory_augmentation_extra_views_only=getattr(
            cfg,
            'temporal_memory_trajectory_augmentation_extra_views_only',
            True,
        ),
        trajectory_augmentation_residual_speed=getattr(
            cfg,
            'temporal_memory_trajectory_augmentation_residual_speed',
            4.0,
        ),
        trajectory_augmentation_min_track_bins=getattr(
            cfg,
            'temporal_memory_trajectory_augmentation_min_track_bins',
            3,
        ),
        cross_video_copy_paste_enabled=getattr(
            cfg,
            'temporal_memory_cross_video_copy_paste_enabled',
            False,
        ),
        cross_video_copy_paste_min_event_count=getattr(
            cfg,
            'temporal_memory_cross_video_copy_paste_min_event_count',
            30000,
        ),
        cross_video_copy_paste_probability=getattr(
            cfg,
            'temporal_memory_cross_video_copy_paste_probability',
            0.25,
        ),
        cross_video_copy_paste_extra_views_only=getattr(
            cfg,
            'temporal_memory_cross_video_copy_paste_extra_views_only',
            True,
        ),
        cross_video_copy_paste_extra_views=getattr(
            cfg,
            'temporal_memory_cross_video_copy_paste_extra_views',
            0,
        ),
        cross_video_copy_paste_min_track_bins=getattr(
            cfg,
            'temporal_memory_cross_video_copy_paste_min_track_bins',
            3,
        ),
        cross_video_copy_paste_collision_radius=getattr(
            cfg,
            'temporal_memory_cross_video_copy_paste_collision_radius',
            1,
        ),
        horizontal_flip_augmentation_enabled=getattr(
            cfg,
            'p15_horizontal_flip_augmentation_enabled',
            False,
        ),
        horizontal_flip_augmentation_probability=getattr(
            cfg,
            'p15_horizontal_flip_augmentation_probability',
            0.50,
        ),
        temporal_phase_offset=getattr(
            cfg,
            'temporal_memory_training_phase_offset',
            0,
        ),
        local_temporal_context_enabled=getattr(
            cfg,
            'temporal_memory_local_temporal_context_enabled',
            False,
        ),
        local_temporal_context_kernel_size=getattr(
            cfg,
            'temporal_memory_local_temporal_context_kernel_size',
            11,
        ),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.temporal_memory_train_workers),
        collate_fn=temporal_memory_collate,
        pin_memory=True,
    )
    density_calibration_enabled = bool(
        getattr(cfg, 'temporal_frame_density_calibration_enabled', False)
    )
    trajectory_enabled = bool(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_enabled', False)
    )
    trajectory_weight = float(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_weight', 0.05)
    )
    trajectory_margin = float(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_margin_logit', 1.0)
    )
    trajectory_min_points = int(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_min_points', 3)
    )
    trajectory_warmup_epochs = int(
        getattr(cfg, 'temporal_frame_trajectory_extrapolation_warmup_epochs', 3)
    )
    if trajectory_enabled:
        if trajectory_weight <= 0.0:
            raise ValueError(
                'TEMPORAL_FRAME.trajectory_extrapolation_weight must be positive.'
            )
        if trajectory_min_points < 2:
            raise ValueError(
                'TEMPORAL_FRAME.trajectory_extrapolation_min_points must be at least 2.'
            )
        if trajectory_warmup_epochs < 0:
            raise ValueError(
                'TEMPORAL_FRAME.trajectory_extrapolation_warmup_epochs must be non-negative.'
            )
    metric_aux_enabled = bool(
        getattr(cfg, 'temporal_memory_metric_aux_enabled', False)
    )
    metric_target_weight = float(
        getattr(cfg, 'temporal_memory_metric_target_weight', 0.01)
    )
    metric_component_weight = float(
        getattr(cfg, 'temporal_memory_metric_component_weight', 0.002)
    )
    metric_warmup_epochs = int(
        getattr(cfg, 'temporal_memory_metric_warmup_epochs', 5)
    )
    metric_spatial_cell_size = int(
        getattr(cfg, 'temporal_memory_metric_spatial_cell_size', 3)
    )
    metric_min_cell_events = int(
        getattr(cfg, 'temporal_memory_metric_min_cell_events', 2)
    )
    metric_component_ratio = float(
        getattr(cfg, 'temporal_memory_metric_component_ratio', 0.01)
    )
    metric_activation_threshold = float(
        getattr(cfg, 'temporal_memory_metric_activation_threshold', 0.70)
    )
    metric_activation_temperature = float(
        getattr(cfg, 'temporal_memory_metric_activation_temperature', 0.10)
    )
    if metric_aux_enabled:
        if metric_target_weight < 0.0 or metric_component_weight < 0.0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric target/component weights must be non-negative.'
            )
        if metric_warmup_epochs < 0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric_warmup_epochs must be non-negative.'
            )
        if metric_spatial_cell_size <= 0 or metric_min_cell_events <= 0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric cell size and minimum events must be positive.'
            )
        if not 0.0 < metric_component_ratio <= 1.0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric_component_ratio must be in (0, 1].'
            )
        if not 0.0 < metric_activation_threshold < 1.0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric_activation_threshold must be in (0, 1).'
            )
        if metric_activation_temperature <= 0.0:
            raise ValueError(
                'TEMPORAL_MEMORY.metric_activation_temperature must be positive.'
            )
    dense_only_enabled = bool(
        getattr(cfg, 'temporal_memory_dense_only_enabled', False)
    )
    dense_only_event_count_cutoff = int(
        getattr(cfg, 'temporal_memory_dense_only_event_count_cutoff', 100000)
    )
    low_density_only_enabled = bool(
        getattr(cfg, 'temporal_memory_low_density_only_enabled', False)
    )
    low_density_only_event_count_cutoff = int(
        getattr(
            cfg,
            'temporal_memory_low_density_only_event_count_cutoff',
            30000,
        )
    )
    max_videos_per_epoch = int(
        getattr(cfg, 'temporal_memory_max_videos_per_epoch', 0)
    )
    mid_density_sampling_enabled = bool(
        getattr(cfg, 'temporal_memory_mid_density_sampling_enabled', False)
    )
    mid_density_min_event_count = int(
        getattr(cfg, 'temporal_memory_mid_density_min_event_count', 30000)
    )
    mid_density_max_event_count = int(
        getattr(cfg, 'temporal_memory_mid_density_max_event_count', 200000)
    )
    mid_density_view_multiplier = int(
        getattr(cfg, 'temporal_memory_mid_density_view_multiplier', 2)
    )
    hard_negative_enabled = bool(
        getattr(cfg, 'temporal_memory_hard_negative_enabled', False)
    )
    hard_negative_weight = float(
        getattr(cfg, 'temporal_memory_hard_negative_weight', 0.02)
    )
    hard_negative_score_floor = float(
        getattr(cfg, 'temporal_memory_hard_negative_score_floor', 0.50)
    )
    motion_sampling_enabled = bool(
        getattr(cfg, 'temporal_memory_motion_sampling_enabled', False)
    )
    motion_sampling_min_event_count = int(
        getattr(cfg, 'temporal_memory_motion_sampling_min_event_count', 30000)
    )
    motion_sampling_min_displacement = float(
        getattr(cfg, 'temporal_memory_motion_sampling_min_displacement', 4.0)
    )
    motion_sampling_probability = float(
        getattr(cfg, 'temporal_memory_motion_sampling_probability', 0.50)
    )
    motion_sampling_extra_views_only = bool(
        getattr(cfg, 'temporal_memory_motion_sampling_extra_views_only', True)
    )
    horizontal_flip_augmentation_enabled = bool(
        getattr(cfg, 'p15_horizontal_flip_augmentation_enabled', False)
    )
    horizontal_flip_augmentation_probability = float(
        getattr(cfg, 'p15_horizontal_flip_augmentation_probability', 0.50)
    )
    if dense_only_enabled and dense_only_event_count_cutoff <= 0:
        raise ValueError(
            'TEMPORAL_MEMORY.dense_only_event_count_cutoff must be positive.'
        )
    if (
        low_density_only_enabled
        and low_density_only_event_count_cutoff <= 0
    ):
        raise ValueError(
            'TEMPORAL_MEMORY.low_density_only_event_count_cutoff must be positive.'
        )
    if dense_only_enabled and low_density_only_enabled:
        raise ValueError(
            'TEMPORAL_MEMORY dense-only and low-density-only routes are exclusive.'
        )
    if max_videos_per_epoch < 0:
        raise ValueError(
            'TEMPORAL_MEMORY.max_videos_per_epoch must be non-negative.'
        )
    if mid_density_sampling_enabled:
        if mid_density_min_event_count < 0:
            raise ValueError(
                'TEMPORAL_MEMORY.mid_density_min_event_count must be non-negative.'
            )
        if mid_density_max_event_count <= mid_density_min_event_count:
            raise ValueError(
                'TEMPORAL_MEMORY.mid_density_max_event_count must exceed the minimum.'
            )
        if mid_density_view_multiplier < 2:
            raise ValueError(
                'TEMPORAL_MEMORY.mid_density_view_multiplier must be at least two.'
            )
    if hard_negative_enabled:
        if hard_negative_weight <= 0.0:
            raise ValueError(
                'TEMPORAL_MEMORY.hard_negative_weight must be positive.'
            )
        if not 0.0 <= hard_negative_score_floor < 1.0:
            raise ValueError(
                'TEMPORAL_MEMORY.hard_negative_score_floor must be in [0, 1).'
            )
    if motion_sampling_min_event_count < 0:
        raise ValueError(
            'TEMPORAL_MEMORY.motion_sampling_min_event_count must be non-negative.'
        )
    if motion_sampling_min_displacement < 0.0:
        raise ValueError(
            'TEMPORAL_MEMORY.motion_sampling_min_displacement must be non-negative.'
        )
    if not 0.0 <= motion_sampling_probability <= 1.0:
        raise ValueError(
            'TEMPORAL_MEMORY.motion_sampling_probability must be in [0, 1].'
        )
    checkpoint_interval = int(getattr(cfg, 'checkpoint_interval', 0))
    if checkpoint_interval < 0:
        raise ValueError('TRAIN.checkpoint_interval must be non-negative.')
    confidence_head_enabled = bool(
        getattr(cfg, 'temporal_frame_confidence_head_enabled', False)
    )
    confidence_only_enabled = bool(
        getattr(cfg, 'temporal_memory_confidence_only_enabled', False)
    )
    advection_flow_only_enabled = bool(
        getattr(cfg, 'temporal_memory_advection_flow_only_enabled', False)
    )
    memory_only_enabled = bool(
        getattr(cfg, 'temporal_memory_memory_only_enabled', False)
    )
    local_temporal_context_enabled = bool(
        getattr(cfg, 'temporal_memory_local_temporal_context_enabled', False)
    )
    local_temporal_context_kernel_size = int(
        getattr(cfg, 'temporal_memory_local_temporal_context_kernel_size', 11)
    )
    local_temporal_context_only_enabled = bool(
        getattr(
            cfg,
            'temporal_memory_local_temporal_context_only_enabled',
            False,
        )
    )
    fine_memory_only_enabled = bool(
        getattr(cfg, 'temporal_memory_fine_memory_only_enabled', False)
    )
    target_center_enabled = bool(
        getattr(cfg, 'temporal_memory_target_center_enabled', False)
    )
    center_memory_enabled = bool(
        getattr(cfg, 'temporal_memory_center_memory_enabled', False)
    )
    center_memory_channels = int(
        getattr(cfg, 'temporal_memory_center_memory_channels', 4)
    )
    center_memory_downsample = int(
        getattr(cfg, 'temporal_memory_center_memory_downsample', 4)
    )
    center_memory_only_enabled = bool(
        getattr(cfg, 'temporal_memory_center_memory_only_enabled', False)
    )
    center_memory_loss_weight = float(
        getattr(cfg, 'temporal_memory_target_center_loss_weight', 0.05)
    )
    center_memory_sigma = float(
        getattr(cfg, 'temporal_memory_target_center_sigma', 2.5)
    )
    center_memory_radius = int(
        getattr(cfg, 'temporal_memory_target_center_radius', 6)
    )
    center_memory_positive_loss_mass = float(
        getattr(
            cfg,
            'temporal_memory_target_center_positive_loss_mass',
            0.20,
        )
    )
    center_memory_max_positive_weight = float(
        getattr(
            cfg,
            'temporal_memory_target_center_max_positive_weight',
            512.0,
        )
    )
    center_memory_empty_loss_weight = float(
        getattr(
            cfg,
            'temporal_memory_target_center_empty_loss_weight',
            0.10,
        )
    )
    target_level_enabled = bool(
        getattr(cfg, 'temporal_memory_target_level_enabled', False)
    )
    target_level_center_loss_weight = float(
        getattr(cfg, 'temporal_memory_target_level_center_loss_weight', 0.025)
    )
    target_level_presence_loss_weight = float(
        getattr(cfg, 'temporal_memory_target_level_presence_loss_weight', 0.010)
    )
    target_level_velocity_loss_weight = float(
        getattr(cfg, 'temporal_memory_target_level_velocity_loss_weight', 0.010)
    )
    target_level_center_sigma = float(
        getattr(cfg, 'temporal_memory_target_level_center_sigma', 2.5)
    )
    target_level_center_radius = int(
        getattr(cfg, 'temporal_memory_target_level_center_radius', 6)
    )
    target_level_positive_loss_mass = float(
        getattr(cfg, 'temporal_memory_target_level_positive_loss_mass', 0.20)
    )
    target_level_max_positive_weight = float(
        getattr(cfg, 'temporal_memory_target_level_max_positive_weight', 512.0)
    )
    target_level_empty_loss_weight = float(
        getattr(cfg, 'temporal_memory_target_level_empty_loss_weight', 0.10)
    )
    target_level_velocity_huber_delta = float(
        getattr(cfg, 'temporal_memory_target_level_velocity_huber_delta', 2.0)
    )
    target_level_downsample = int(
        getattr(cfg, 'temporal_memory_target_level_downsample', 4)
    )
    advection_fast_motion_threshold = float(
        getattr(cfg, 'temporal_memory_advection_fast_motion_threshold', 0.0)
    )
    advection_fast_motion_weight = float(
        getattr(cfg, 'temporal_memory_advection_fast_motion_weight', 1.0)
    )
    if confidence_only_enabled and not confidence_head_enabled:
        raise ValueError(
            'TEMPORAL_MEMORY.confidence_only_enabled requires '
            'TEMPORAL_FRAME.confidence_head_enabled=true.'
        )
    if sum((
        bool(confidence_only_enabled),
        bool(fine_memory_only_enabled),
        bool(center_memory_only_enabled),
        bool(advection_flow_only_enabled),
        bool(memory_only_enabled),
        bool(local_temporal_context_only_enabled),
    )) > 1:
        raise ValueError(
            'Only one isolated adapter training mode may be enabled at a time.'
        )
    confidence_calibration_weight = float(
        getattr(cfg, 'temporal_frame_confidence_calibration_weight', 0.1)
    )
    if confidence_head_enabled and confidence_calibration_weight <= 0.0:
        raise ValueError(
            'TEMPORAL_FRAME.confidence_calibration_weight must be positive.'
        )
    temporal_attention_enabled = bool(
        getattr(cfg, 'temporal_memory_temporal_attention_enabled', False)
    )
    temporal_attention_output_init_std = float(
        getattr(cfg, 'temporal_memory_attention_output_init_std', 0.0)
    )
    if temporal_attention_output_init_std < 0.0:
        raise ValueError(
            'TEMPORAL_MEMORY.attention_output_init_std must be non-negative.'
        )
    temporal_attention_relative_bias_enabled = bool(
        getattr(cfg, 'temporal_memory_attention_relative_bias_enabled', False)
    )
    temporal_attention_relative_bias_max_distance = int(
        getattr(cfg, 'temporal_memory_attention_relative_bias_max_distance', 8)
    )
    if (
        temporal_attention_relative_bias_enabled
        and temporal_attention_relative_bias_max_distance <= 0
    ):
        raise ValueError(
            'TEMPORAL_MEMORY.attention_relative_bias_max_distance must be positive.'
        )
    advection_alignment_enabled = bool(
        getattr(cfg, 'temporal_memory_advection_alignment_enabled', False)
    )
    advection_alignment_loss_weight = float(
        getattr(cfg, 'temporal_memory_advection_alignment_loss_weight', 0.05)
    )
    advection_max_flow = float(
        getattr(cfg, 'temporal_memory_advection_max_flow', 0.0)
    )
    advection_target_flow_enabled = bool(
        getattr(cfg, 'temporal_memory_advection_target_flow_enabled', False)
    )
    advection_target_flow_weight = float(
        getattr(cfg, 'temporal_memory_advection_target_flow_weight', 0.5)
    )
    advection_target_flow_huber_delta = float(
        getattr(cfg, 'temporal_memory_advection_target_flow_huber_delta', 1.0)
    )
    advection_trajectory_flow_enabled = bool(
        getattr(cfg, 'temporal_memory_advection_trajectory_flow_enabled', False)
    )
    advection_trajectory_flow_weight = float(
        getattr(cfg, 'temporal_memory_advection_trajectory_flow_weight', 0.15)
    )
    advection_trajectory_flow_max_hop = int(
        getattr(cfg, 'temporal_memory_advection_trajectory_flow_max_hop', 2)
    )
    fine_temporal_memory_enabled = bool(
        getattr(cfg, 'temporal_memory_fine_temporal_memory_enabled', False)
    )
    fine_advection_max_flow = float(
        getattr(cfg, 'temporal_memory_fine_advection_max_flow', 0.0)
    )
    fine_target_flow_enabled = bool(
        getattr(cfg, 'temporal_memory_fine_target_flow_enabled', False)
    )
    fine_target_flow_weight = float(
        getattr(cfg, 'temporal_memory_fine_target_flow_weight', 0.5)
    )
    if (
        advection_alignment_enabled
        and not advection_flow_only_enabled
        and not memory_only_enabled
        and not local_temporal_context_only_enabled
        and advection_alignment_loss_weight <= 0.0
    ):
        raise ValueError(
            'TEMPORAL_MEMORY.advection_alignment_loss_weight must be positive.'
        )
    if advection_max_flow < 0.0:
        raise ValueError('TEMPORAL_MEMORY.advection_max_flow must be non-negative.')
    if advection_target_flow_enabled and not advection_alignment_enabled:
        raise ValueError(
            'TEMPORAL_MEMORY.target_flow_enabled requires advection alignment.'
        )
    if advection_target_flow_enabled and advection_target_flow_weight <= 0.0:
        raise ValueError('TEMPORAL_MEMORY.target_flow_weight must be positive.')
    if advection_target_flow_enabled and advection_target_flow_huber_delta <= 0.0:
        raise ValueError('TEMPORAL_MEMORY.target_flow_huber_delta must be positive.')
    if advection_flow_only_enabled and not advection_alignment_enabled:
        raise ValueError(
            'Flow-only mode requires TEMPORAL_MEMORY.advection_alignment_enabled=true.'
        )
    if advection_flow_only_enabled and not advection_target_flow_enabled:
        raise ValueError(
            'Flow-only mode requires target-centroid flow supervision.'
        )
    if memory_only_enabled and not temporal_attention_enabled:
        raise ValueError(
            'Memory-only mode requires temporal attention in the M26 backbone.'
        )
    if memory_only_enabled and advection_target_flow_enabled:
        raise ValueError(
            'Memory-only mode freezes the flow head, so '
            'TEMPORAL_MEMORY.advection_target_flow_enabled must be false.'
        )
    if memory_only_enabled and advection_alignment_loss_weight != 0.0:
        raise ValueError(
            'Memory-only mode must set advection consistency weight to zero.'
        )
    if memory_only_enabled and advection_trajectory_flow_enabled:
        raise ValueError(
            'Memory-only mode does not combine the rejected M27 trajectory loss.'
        )
    if (
        local_temporal_context_kernel_size <= 0
        or local_temporal_context_kernel_size % 2 == 0
    ):
        raise ValueError(
            'TEMPORAL_MEMORY.local_temporal_context_kernel_size must be a '
            'positive odd integer.'
        )
    if local_temporal_context_only_enabled and not local_temporal_context_enabled:
        raise ValueError(
            'Local-context-only mode requires '
            'temporal_memory_local_temporal_context_enabled=true.'
        )
    if local_temporal_context_only_enabled and (
        confidence_head_enabled
        or target_center_enabled
        or metric_aux_enabled
        or trajectory_enabled
        or hard_negative_enabled
        or advection_target_flow_enabled
        or advection_trajectory_flow_enabled
        or advection_alignment_loss_weight != 0.0
    ):
        raise ValueError(
            'Local-context-only mode is restricted to event BCE and must '
            'disable auxiliary heads and losses.'
        )
    if advection_flow_only_enabled and advection_alignment_loss_weight != 0.0:
        raise ValueError(
            'Flow-only mode must set advection consistency weight to zero.'
        )
    if advection_flow_only_enabled and advection_trajectory_flow_enabled:
        raise ValueError(
            'Flow-only mode does not combine the rejected M27 trajectory loss.'
        )
    if advection_fast_motion_threshold < 0.0:
        raise ValueError('Fast-motion threshold must be non-negative.')
    if advection_fast_motion_weight < 1.0:
        raise ValueError('Fast-motion weight must be at least one.')
    if advection_trajectory_flow_enabled and not advection_alignment_enabled:
        raise ValueError(
            'TEMPORAL_MEMORY.trajectory_flow_enabled requires advection alignment.'
        )
    if advection_trajectory_flow_enabled and advection_trajectory_flow_weight <= 0.0:
        raise ValueError(
            'TEMPORAL_MEMORY.trajectory_flow_weight must be positive.'
        )
    if advection_trajectory_flow_enabled and advection_trajectory_flow_max_hop < 2:
        raise ValueError(
            'TEMPORAL_MEMORY.trajectory_flow_max_hop must be at least two.'
        )
    if fine_advection_max_flow < 0.0:
        raise ValueError(
            'TEMPORAL_MEMORY.fine_advection_max_flow must be non-negative.'
        )
    if fine_target_flow_enabled and not fine_temporal_memory_enabled:
        raise ValueError(
            'TEMPORAL_MEMORY.fine_target_flow_enabled requires fine temporal memory.'
        )
    if fine_temporal_memory_enabled and fine_advection_max_flow <= 0.0:
        raise ValueError(
            'TEMPORAL_MEMORY.fine_advection_max_flow must be positive when enabled.'
        )
    if fine_target_flow_enabled and fine_target_flow_weight <= 0.0:
        raise ValueError(
            'TEMPORAL_MEMORY.fine_target_flow_weight must be positive.'
        )
    if fine_memory_only_enabled and not fine_temporal_memory_enabled:
        raise ValueError(
            'TEMPORAL_MEMORY.fine_memory_only_enabled requires fine temporal memory.'
        )
    if fine_memory_only_enabled and advection_target_flow_enabled:
        raise ValueError(
            'Fine-memory-only mode freezes the H/8 flow, so '
            'TEMPORAL_MEMORY.advection_target_flow_enabled must be false.'
        )
    if fine_memory_only_enabled and advection_trajectory_flow_enabled:
        raise ValueError(
            'Fine-memory-only mode freezes the H/8 flow, so '
            'TEMPORAL_MEMORY.advection_trajectory_flow_enabled must be false.'
        )
    if center_memory_enabled and not target_center_enabled:
        raise ValueError(
            'TEMPORAL_MEMORY.center_memory_enabled requires '
            'temporal_memory_target_center_enabled=true.'
        )
    if center_memory_only_enabled and not center_memory_enabled:
        raise ValueError(
            'TEMPORAL_MEMORY.center_memory_only_enabled requires center memory.'
        )
    if target_center_enabled and confidence_head_enabled:
        raise ValueError(
            'Target-centre memory and confidence calibration cannot be enabled together.'
        )
    if target_level_enabled and (
        target_center_enabled
        or confidence_head_enabled
        or confidence_only_enabled
        or center_memory_only_enabled
        or fine_memory_only_enabled
        or advection_flow_only_enabled
        or memory_only_enabled
        or local_temporal_context_only_enabled
    ):
        raise ValueError(
            'M91 target-level training must be the joint event+BCE path; '
            'disable competing auxiliary-only modes.'
        )
    if center_memory_channels <= 0 or center_memory_downsample < 2:
        raise ValueError(
            'Center-memory channels must be positive and downsample at least two.'
        )
    if target_center_enabled and center_memory_loss_weight <= 0.0:
        raise ValueError('Target-centre loss weight must be positive.')
    if center_memory_sigma <= 0.0 or center_memory_radius <= 0:
        raise ValueError('Target-centre sigma and radius must be positive.')
    if not 0.0 < center_memory_positive_loss_mass < 1.0:
        raise ValueError('Target-centre positive loss mass must be in (0, 1).')
    if center_memory_max_positive_weight < 1.0:
        raise ValueError('Target-centre maximum positive weight must be at least one.')
    if not 0.0 <= center_memory_empty_loss_weight <= 1.0:
        raise ValueError('Target-centre empty loss weight must be in [0, 1].')
    if target_level_enabled:
        if (
            target_level_center_loss_weight <= 0.0
            or target_level_presence_loss_weight <= 0.0
            or target_level_velocity_loss_weight <= 0.0
        ):
            raise ValueError('M91 target-level loss weights must be positive.')
        if target_level_center_sigma <= 0.0 or target_level_center_radius <= 0:
            raise ValueError('M91 target-level centre geometry is invalid.')
        if not 0.0 < target_level_positive_loss_mass < 1.0:
            raise ValueError('M91 target-level positive loss mass must be in (0, 1).')
        if target_level_max_positive_weight < 1.0:
            raise ValueError('M91 target-level max positive weight must be at least one.')
        if not 0.0 <= target_level_empty_loss_weight <= 1.0:
            raise ValueError('M91 target-level empty loss weight must be in [0, 1].')
        if target_level_velocity_huber_delta <= 0.0:
            raise ValueError('M91 target-level velocity huber delta must be positive.')
        if target_level_downsample <= 0:
            raise ValueError('M91 target-level downsample must be positive.')
    model = BidirectionalTemporalMemoryNet(
        input_channels=int(cfg.temporal_memory_context_bins) * 2,
        width=int(cfg.temporal_memory_width),
        density_calibration_enabled=density_calibration_enabled,
        confidence_head_enabled=confidence_head_enabled,
        temporal_attention_enabled=temporal_attention_enabled,
        temporal_attention_output_init_std=temporal_attention_output_init_std,
        temporal_attention_relative_bias_enabled=temporal_attention_relative_bias_enabled,
        temporal_attention_relative_bias_max_distance=(
            temporal_attention_relative_bias_max_distance
        ),
        advection_alignment_enabled=advection_alignment_enabled,
        advection_max_flow=advection_max_flow,
        fine_temporal_memory_enabled=fine_temporal_memory_enabled,
        fine_advection_max_flow=fine_advection_max_flow,
        local_temporal_context_enabled=local_temporal_context_enabled,
        local_temporal_context_kernel_size=local_temporal_context_kernel_size,
        target_center_enabled=target_center_enabled,
        target_level_enabled=target_level_enabled,
        target_level_downsample=target_level_downsample,
        center_memory_enabled=center_memory_enabled,
        center_memory_channels=center_memory_channels,
        center_memory_downsample=center_memory_downsample,
    ).to(device)
    initialized_from = load_p23_base_weights(
        model,
        cfg.temporal_memory_init_model_path,
        cfg.temporal_memory_context_bins,
        cfg.temporal_memory_width,
        density_calibration_enabled=density_calibration_enabled,
        confidence_head_enabled=confidence_head_enabled,
        target_center_enabled=target_center_enabled,
        target_level_enabled=target_level_enabled,
        center_memory_enabled=center_memory_enabled,
        local_temporal_context_enabled=local_temporal_context_enabled,
    )
    if confidence_only_enabled:
        confidence_parameter_ids = {
            id(parameter) for parameter in model.base.confidence_head.parameters()
        }
        for parameter in model.parameters():
            parameter.requires_grad = id(parameter) in confidence_parameter_ids
    elif fine_memory_only_enabled:
        fine_parameter_ids = set()
        for module in (
            model.fine_forward_memory,
            model.fine_backward_memory,
            model.fine_memory_projection,
            model.fine_flow_head,
        ):
            fine_parameter_ids.update(id(parameter) for parameter in module.parameters())
        for parameter in model.parameters():
            parameter.requires_grad = id(parameter) in fine_parameter_ids
    elif center_memory_only_enabled:
        center_parameter_ids = set()
        for module in (
            model.base.target_center_head,
            model.base.target_center_residual,
            model.center_memory_projection_in,
            model.center_memory_forward,
            model.center_memory_backward,
            model.center_memory_projection_out,
            model.center_event_projection,
        ):
            center_parameter_ids.update(
                id(parameter) for parameter in module.parameters()
            )
        for parameter in model.parameters():
            parameter.requires_grad = id(parameter) in center_parameter_ids
    elif advection_flow_only_enabled:
        flow_parameter_ids = {
            id(parameter) for parameter in model.flow_head.parameters()
        }
        for parameter in model.parameters():
            parameter.requires_grad = id(parameter) in flow_parameter_ids
    elif memory_only_enabled:
        memory_parameter_ids = set()
        for module in (
            model.forward_memory,
            model.backward_memory,
            model.memory_projection,
            model.temporal_attn,
        ):
            memory_parameter_ids.update(id(parameter) for parameter in module.parameters())
        for parameter in model.parameters():
            parameter.requires_grad = id(parameter) in memory_parameter_ids
    elif local_temporal_context_only_enabled:
        local_temporal_context_parameter_ids = {
            id(parameter)
            for parameter in model.base.local_temporal_context_adapter.parameters()
        }
        for parameter in model.parameters():
            parameter.requires_grad = (
                id(parameter) in local_temporal_context_parameter_ids
            )
    optimizer = build_optimizer(
        model,
        cfg,
        confidence_only_enabled=confidence_only_enabled,
        fine_memory_only_enabled=fine_memory_only_enabled,
        center_memory_only_enabled=center_memory_only_enabled,
        advection_flow_only_enabled=advection_flow_only_enabled,
        memory_only_enabled=memory_only_enabled,
        local_temporal_context_only_enabled=local_temporal_context_only_enabled,
    )
    if advection_flow_only_enabled:
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group['params']
        }
        flow_parameter_ids = {
            id(parameter) for parameter in model.flow_head.parameters()
        }
        if optimizer_parameter_ids != flow_parameter_ids:
            raise RuntimeError('M52 optimizer must contain exactly flow_head parameters.')
    if memory_only_enabled:
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group['params']
        }
        expected_parameter_ids = {
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        if optimizer_parameter_ids != expected_parameter_ids:
            raise RuntimeError(
                'Memory-only optimizer must contain exactly ConvGRU, projection, '
                'and temporal-attention parameters.'
            )
    if local_temporal_context_only_enabled:
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group['params']
        }
        expected_parameter_ids = {
            id(parameter)
            for parameter in model.base.local_temporal_context_adapter.parameters()
        }
        if optimizer_parameter_ids != expected_parameter_ids:
            raise RuntimeError(
                'Local-context-only optimizer must contain only the local '
                'temporal-context adapter.'
            )
    scheduler = build_scheduler(optimizer, cfg)

    print('random seed:{}'.format(cfg.seed))
    print('run directory:', run_dir)
    print('config overrides:', ', '.join(cfg.config_overrides) or '(none)')
    print('temporal-memory model:', memory_config_summary(cfg))
    print('training videos:', len(dataset.file_paths))
    print('training sequences per epoch:', len(dataset))
    print(
        'dense sequence sampling: enabled={}, cutoff={}, multiplier={}, '
        'dense_videos={}, extra_views={}'.format(
            dataset.dense_sampling_enabled,
            dataset.dense_event_count_cutoff,
            dataset.dense_view_multiplier,
            dataset.dense_video_count,
            dataset.extra_dense_views,
        )
    )
    print(
        'mid-density sequence sampling: enabled={}, event_count > {} and <= {}, '
        'multiplier={}, videos={}, extra_views={}'.format(
            mid_density_sampling_enabled,
            mid_density_min_event_count,
            mid_density_max_event_count,
            mid_density_view_multiplier,
            dataset.mid_density_video_count,
            dataset.extra_mid_density_views,
        )
    )
    print(
        'dense-only training: enabled={}, event_count > {}, source_videos={}, '
        'max_videos_per_epoch={}, active_videos={}'.format(
            dense_only_enabled,
            dense_only_event_count_cutoff,
            dataset.source_video_count,
            max_videos_per_epoch,
            dataset.active_video_count,
        )
    )
    print(
        'low-density-only training: enabled={}, event_count <= {}, '
        'source_videos={}'.format(
            low_density_only_enabled,
            low_density_only_event_count_cutoff,
            dataset.source_video_count,
        )
    )
    print(
        'hard-negative score loss: enabled={}, weight={}, score_floor={}'.format(
            hard_negative_enabled,
            hard_negative_weight,
            hard_negative_score_floor,
        )
    )
    print(
        'motion sampling: enabled={}, min_event_count={}, min_displacement={}, '
        'probability={}, extra_views_only={}, fast_bin_count={}'.format(
            motion_sampling_enabled,
            motion_sampling_min_event_count,
            motion_sampling_min_displacement,
            motion_sampling_probability,
            motion_sampling_extra_views_only,
            dataset.fast_bin_count,
        )
    )
    print(
        'trajectory augmentation: enabled={}, min_event_count={}, '
        'probability={}, extra_views_only={}, residual_speed={}, augmented_views={}'.format(
            dataset.trajectory_augmentation_enabled,
            dataset.trajectory_augmentation_min_event_count,
            dataset.trajectory_augmentation_probability,
            dataset.trajectory_augmentation_extra_views_only,
            dataset.trajectory_augmentation_residual_speed,
            dataset.trajectory_augmented_views,
        )
    )
    print(
        'M93 cross-video copy-paste: enabled={}, min_event_count={}, '
        'probability={}, extra_views_only={}, extra_views_per_video={}, '
        'eligible_videos={}, added_views={}, pasted_views={}'.format(
            dataset.cross_video_copy_paste_enabled,
            dataset.cross_video_copy_paste_min_event_count,
            dataset.cross_video_copy_paste_probability,
            dataset.cross_video_copy_paste_extra_views_only,
            dataset.cross_video_copy_paste_extra_views,
            dataset.copy_paste_video_count,
            dataset.copy_paste_extra_views_total,
            dataset.cross_video_copy_paste_views,
        )
    )
    print(
        'horizontal flip augmentation: enabled={}, probability={}'.format(
            horizontal_flip_augmentation_enabled,
            horizontal_flip_augmentation_probability,
        )
    )
    if 'temporal_memory' in torch.load(initialized_from, map_location='cpu'):
        print('initialized full temporal-memory weights from:', initialized_from)
    else:
        print('initialized P23 base weights from:', initialized_from)
    print('learning-rate scheduler:', cfg.scheduler)
    if advection_alignment_enabled:
        print(
            'M25/M26 advection alignment: enabled '
            '(consistency_weight={:.3f}, max_flow={:.3f}, '
                'target_flow_enabled={}, target_flow_weight={:.3f})'.format(
                advection_alignment_loss_weight,
                advection_max_flow,
                advection_target_flow_enabled,
                advection_target_flow_weight,
            )
        )
        if advection_trajectory_flow_enabled:
            print(
                'M27 composed trajectory supervision: enabled '
                '(weight={:.3f}, max_hop={})'.format(
                    advection_trajectory_flow_weight,
                    advection_trajectory_flow_max_hop,
                )
            )
    else:
        print('M25 advection alignment: disabled')
    if fine_temporal_memory_enabled:
        print(
            'M35 fine H/4 memory: enabled '
            '(max_flow={:.3f}, target_flow_enabled={}, target_flow_weight={:.3f})'.format(
                fine_advection_max_flow,
                fine_target_flow_enabled,
                fine_target_flow_weight,
            )
        )
    else:
        print('M35 fine H/4 memory: disabled')
    if confidence_only_enabled:
        print('confidence calibration mode: backbone and memory frozen')
    if fine_memory_only_enabled:
        print('fine-memory-only mode: M26 backbone and original memory frozen')
    if advection_flow_only_enabled:
        print(
            'M52 flow-only mode: M26 representation frozen '
            '(fast_motion_threshold={}, fast_motion_weight={})'.format(
                advection_fast_motion_threshold,
                advection_fast_motion_weight,
            )
        )
    if memory_only_enabled:
        print(
            'M69 memory-only mode: base decoder and flow frozen; ConvGRU, '
            'memory projection, and attention train on the configured '
            'sequence length.'
        )
    if local_temporal_context_only_enabled:
        print(
            'M72 local-context-only mode: M26 and all auxiliary branches '
            'frozen; only the zero-attached local temporal-context adapter trains.'
        )

    best_loss = float('inf')
    best_epoch = None
    for epoch in range(int(cfg.epochs)):
        dataset.set_epoch(epoch)
        print(
            'epoch {} training videos: {}'.format(
                epoch,
                dataset.active_video_count,
            )
        )
        print(
            'epoch {} motion sampling: fast_bin_count={}, selected_views_so_far={}'.format(
                epoch,
                dataset.fast_bin_count,
                dataset.motion_selected_views,
            )
        )
        print(
            'epoch {} trajectory augmentation: selected_views_so_far={}'.format(
                epoch,
                dataset.trajectory_augmented_views,
            )
        )
        print(
            'epoch {} horizontal flip augmentation: selected_views_so_far={}'.format(
                epoch,
                dataset.horizontal_flip_views,
            )
        )
        if confidence_only_enabled:
            # Keep the released M5 representation deterministic and train only
            # the newly attached head.
            model.eval()
            model.base.confidence_head.train()
        elif fine_memory_only_enabled:
            # Preserve the calibrated M26 path while the zero-attached H/4
            # branch learns its residual and flow in training mode.
            model.eval()
            model.fine_forward_memory.train()
            model.fine_backward_memory.train()
            model.fine_memory_projection.train()
            model.fine_flow_head.train()
        elif center_memory_only_enabled:
            # M48 must leave every M26 module, including normalization state,
            # untouched while its centre-specific residual learns.
            model.eval()
            for module in (
                model.base.target_center_head,
                model.base.target_center_residual,
                model.center_memory_projection_in,
                model.center_memory_forward,
                model.center_memory_backward,
                model.center_memory_projection_out,
                model.center_event_projection,
            ):
                module.train()
        elif advection_flow_only_enabled:
            model.eval()
            model.flow_head.train()
        elif memory_only_enabled:
            model.eval()
            model.forward_memory.train()
            model.backward_memory.train()
            model.memory_projection.train()
            model.temporal_attn.train()
        elif local_temporal_context_only_enabled:
            model.eval()
            model.base.local_temporal_context_adapter.train()
        else:
            model.train()
        loss_sum = 0.0
        positive_fraction_sum = 0.0
        positive_weight_sum = 0.0
        trajectory_loss_sum = 0.0
        confidence_loss_sum = 0.0
        metric_target_loss_sum = 0.0
        metric_component_loss_sum = 0.0
        hard_negative_loss_sum = 0.0
        hard_negative_fraction_sum = 0.0
        advection_loss_sum = 0.0
        target_flow_loss_sum = 0.0
        target_flow_pair_sum = 0
        target_flow_fast_pair_sum = 0
        fine_target_flow_loss_sum = 0.0
        fine_target_flow_pair_sum = 0
        trajectory_flow_loss_sum = 0.0
        trajectory_flow_pair_sum = 0
        target_center_loss_sum = 0.0
        target_level_center_loss_sum = 0.0
        target_level_presence_loss_sum = 0.0
        target_level_velocity_loss_sum = 0.0
        target_level_velocity_pair_sum = 0
        batch_count = 0
        pbar = tqdm.tqdm(
            dataloader,
            desc='Epoch: {}'.format(epoch),
            unit='Sequence',
            position=0,
            leave=True,
        )
        for batch in pbar:
            frames = batch['frames'].to(device, non_blocking=True).unsqueeze(0)
            event_time_indices = batch['event_time_indices'].to(
                device,
                non_blocking=True,
            )
            event_y = batch['event_y'].to(device, non_blocking=True)
            event_x = batch['event_x'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            target_ids = batch['target_ids'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            model_output = model(
                frames,
                return_target_center_logits=target_center_enabled,
                return_target_level_outputs=target_level_enabled,
            )
            target_level_center_logits = None
            target_level_presence_logits = None
            target_level_velocity_maps = None
            if target_level_enabled:
                (
                    logit_maps,
                    target_level_center_logits,
                    target_level_presence_logits,
                    target_level_velocity_maps,
                ) = model_output
                logit_maps = logit_maps.squeeze(0)
                target_level_center_logits = target_level_center_logits.squeeze(0)
                target_level_presence_logits = target_level_presence_logits.squeeze(0)
                target_level_velocity_maps = target_level_velocity_maps.squeeze(0)
                confidence_logit_maps = None
                target_center_logits = None
            elif target_center_enabled:
                logit_maps, target_center_logits = model_output
                logit_maps = logit_maps.squeeze(0)
                target_center_logits = target_center_logits.squeeze(0)
                confidence_logit_maps = None
            elif confidence_head_enabled:
                logit_maps, confidence_logit_maps = model_output
                logit_maps = logit_maps.squeeze(0)
                confidence_logit_maps = confidence_logit_maps.squeeze(0)
                target_center_logits = None
            else:
                logit_maps = model_output.squeeze(0)
                target_center_logits = None
            event_logits = logit_maps[
                event_time_indices,
                0,
                event_y,
                event_x,
            ]
            if advection_flow_only_enabled:
                loss = event_logits.sum() * 0.0
                diagnostics = {
                    'positive_fraction': 0.0,
                    'mean_positive_weight': 0.0,
                }
            else:
                loss, diagnostics = frame_balanced_event_bce(
                    event_logits,
                    labels,
                    event_time_indices,
                    target_positive_loss_mass=(
                        cfg.temporal_memory_target_positive_loss_mass
                    ),
                    max_positive_weight=cfg.temporal_memory_max_positive_weight,
                )
            hard_negative_loss = event_logits.sum() * 0.0
            hard_negative_diagnostics = {'hard_negative_fraction': 0.0}
            if hard_negative_enabled:
                hard_negative_loss, hard_negative_diagnostics = (
                    hard_negative_score_loss(
                        event_logits,
                        labels,
                        score_floor=hard_negative_score_floor,
                    )
                )
                loss = loss + hard_negative_weight * hard_negative_loss
            confidence_loss = event_logits.sum() * 0.0
            if confidence_head_enabled:
                event_confidence_logits = confidence_logit_maps[
                    event_time_indices,
                    0,
                    event_y,
                    event_x,
                ]
                confidence_loss = confidence_calibration_loss(
                    event_confidence_logits,
                    event_logits,
                    labels,
                    hard_target=True,
                )
                loss = loss + confidence_calibration_weight * confidence_loss
            target_center_loss = event_logits.sum() * 0.0
            if target_center_enabled:
                target_heatmaps = build_target_center_heatmaps(
                    event_x,
                    event_y,
                    labels,
                    target_ids,
                    event_time_indices,
                    batch_size=target_center_logits.shape[0],
                    height=target_center_logits.shape[2],
                    width=target_center_logits.shape[3],
                    sigma=center_memory_sigma,
                    radius=center_memory_radius,
                )
                target_center_loss, _ = target_center_heatmap_loss(
                    target_center_logits,
                    target_heatmaps,
                    target_positive_loss_mass=center_memory_positive_loss_mass,
                    max_positive_weight=center_memory_max_positive_weight,
                    empty_loss_weight=center_memory_empty_loss_weight,
                )
                loss = loss + center_memory_loss_weight * target_center_loss
            target_level_center_loss = event_logits.sum() * 0.0
            target_level_presence_loss = event_logits.sum() * 0.0
            target_level_velocity_loss = event_logits.sum() * 0.0
            target_level_velocity_stats = {'pair_count': 0, 'mean_motion': 0.0}
            if target_level_enabled:
                target_level_height = target_level_center_logits.shape[2]
                target_level_width = target_level_center_logits.shape[3]
                # M91 predicts at H/4 to keep this training-only objective
                # inexpensive.  Rescale labelled centres into that grid so
                # both heatmap and velocity supervision retain their geometry.
                target_level_x = torch.round(
                    event_x.float()
                    * float(target_level_width - 1)
                    / max(int(cfg.res[0]) - 1, 1)
                ).long()
                target_level_y = torch.round(
                    event_y.float()
                    * float(target_level_height - 1)
                    / max(int(cfg.res[1]) - 1, 1)
                ).long()
                target_level_scale = min(
                    float(target_level_width) / float(cfg.res[0]),
                    float(target_level_height) / float(cfg.res[1]),
                )
                target_level_heatmaps = build_target_center_heatmaps(
                    target_level_x,
                    target_level_y,
                    labels,
                    target_ids,
                    event_time_indices,
                    batch_size=target_level_center_logits.shape[0],
                    height=target_level_height,
                    width=target_level_width,
                    sigma=max(0.5, target_level_center_sigma * target_level_scale),
                    radius=max(1, int(round(
                        target_level_center_radius * target_level_scale
                    ))),
                )
                target_level_center_loss, _ = target_center_heatmap_loss(
                    target_level_center_logits,
                    target_level_heatmaps,
                    target_positive_loss_mass=target_level_positive_loss_mass,
                    max_positive_weight=target_level_max_positive_weight,
                    empty_loss_weight=target_level_empty_loss_weight,
                )
                target_level_presence_loss, _ = target_level_presence_loss_fn(
                    target_level_presence_logits,
                    event_time_indices,
                    labels,
                    target_ids,
                )
                target_level_velocity_loss, target_level_velocity_stats = (
                    target_level_velocity_loss_fn(
                        target_level_velocity_maps,
                        event_time_indices,
                        target_level_x,
                        target_level_y,
                        labels,
                        target_ids,
                        huber_delta=target_level_velocity_huber_delta,
                    )
                )
                loss = loss + (
                    target_level_center_loss_weight * target_level_center_loss
                    + target_level_presence_loss_weight * target_level_presence_loss
                    + target_level_velocity_loss_weight * target_level_velocity_loss
                )
            metric_target_loss = event_logits.sum() * 0.0
            metric_component_loss = event_logits.sum() * 0.0
            if metric_aux_enabled and epoch >= metric_warmup_epochs:
                event_scores = torch.sigmoid(event_logits)
                event_locations = torch.stack(
                    (
                        torch.zeros_like(event_x),
                        event_x,
                        event_y,
                        event_time_indices * int(cfg.temporal_memory_bin_size) + 1,
                    ),
                    dim=1,
                )
                if metric_target_weight > 0.0:
                    metric_target_loss, _, _ = target_frame_activation_loss(
                        event_scores,
                        labels,
                        target_ids,
                        event_locations,
                        int(cfg.temporal_memory_bin_size),
                        metric_activation_threshold,
                        metric_activation_temperature,
                    )
                    loss = loss + metric_target_weight * metric_target_loss
                if metric_component_weight > 0.0:
                    metric_component_loss, _, _ = component_hard_negative_loss(
                        event_scores,
                        labels,
                        event_locations,
                        metric_spatial_cell_size,
                        int(cfg.temporal_memory_bin_size),
                        metric_min_cell_events,
                        metric_component_ratio,
                        metric_activation_threshold,
                        metric_activation_temperature,
                    )
                    loss = loss + metric_component_weight * metric_component_loss
            trajectory_loss = event_logits.sum() * 0.0
            if trajectory_enabled and epoch >= trajectory_warmup_epochs:
                trajectory_loss, trajectory_stats = (
                    trajectory_extrapolation_loss_memory(
                        logit_maps,
                        event_time_indices,
                        event_x,
                        event_y,
                        labels,
                        target_ids,
                        min_known_points=trajectory_min_points,
                        margin_logit=trajectory_margin,
                    )
                )
                loss = loss + trajectory_weight * trajectory_loss
            advection_loss = event_logits.sum() * 0.0
            if (
                advection_alignment_enabled
                and not advection_flow_only_enabled
                and not memory_only_enabled
                and not local_temporal_context_only_enabled
            ):
                advection_loss = model._last_advection_consistency_loss
                loss = loss + advection_alignment_loss_weight * advection_loss
            target_flow_loss = event_logits.sum() * 0.0
            target_flow_stats = {'pair_count': 0, 'target_motion_mean': 0.0}
            if advection_target_flow_enabled:
                target_flow_loss, target_flow_stats = target_centroid_flow_loss(
                    model._last_advection_forward_flows,
                    event_time_indices,
                    event_x,
                    event_y,
                    labels,
                    target_ids,
                    input_width=int(cfg.res[0]),
                    input_height=int(cfg.res[1]),
                    huber_delta=advection_target_flow_huber_delta,
                    fast_motion_threshold=(
                        advection_fast_motion_threshold
                        if advection_flow_only_enabled else 0.0
                    ),
                    fast_motion_weight=(
                        advection_fast_motion_weight
                        if advection_flow_only_enabled else 1.0
                    ),
                )
                loss = loss + advection_target_flow_weight * target_flow_loss
            fine_target_flow_loss = event_logits.sum() * 0.0
            fine_target_flow_stats = {'pair_count': 0, 'target_motion_mean': 0.0}
            if fine_target_flow_enabled:
                fine_target_flow_loss, fine_target_flow_stats = target_centroid_flow_loss(
                    model._last_fine_advection_forward_flows,
                    event_time_indices,
                    event_x,
                    event_y,
                    labels,
                    target_ids,
                    input_width=int(cfg.res[0]),
                    input_height=int(cfg.res[1]),
                    huber_delta=advection_target_flow_huber_delta,
                )
                loss = loss + fine_target_flow_weight * fine_target_flow_loss
            trajectory_flow_loss = event_logits.sum() * 0.0
            trajectory_flow_stats = {'pair_count': 0, 'target_motion_mean': 0.0}
            if advection_trajectory_flow_enabled:
                trajectory_flow_loss, trajectory_flow_stats = (
                    target_centroid_trajectory_flow_loss(
                        model._last_advection_forward_flows,
                        event_time_indices,
                        event_x,
                        event_y,
                        labels,
                        target_ids,
                        input_width=int(cfg.res[0]),
                        input_height=int(cfg.res[1]),
                        huber_delta=advection_target_flow_huber_delta,
                        max_hop=advection_trajectory_flow_max_hop,
                    )
                )
                loss = loss + advection_trajectory_flow_weight * trajectory_flow_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            loss_sum += float(loss.detach().item())
            positive_fraction_sum += diagnostics['positive_fraction']
            positive_weight_sum += diagnostics['mean_positive_weight']
            trajectory_loss_sum += float(trajectory_loss.detach().item())
            confidence_loss_sum += float(confidence_loss.detach().item())
            metric_target_loss_sum += float(metric_target_loss.detach().item())
            metric_component_loss_sum += float(metric_component_loss.detach().item())
            hard_negative_loss_sum += float(hard_negative_loss.detach().item())
            hard_negative_fraction_sum += hard_negative_diagnostics[
                'hard_negative_fraction'
            ]
            advection_loss_sum += float(advection_loss.detach().item())
            target_flow_loss_sum += float(target_flow_loss.detach().item())
            target_flow_pair_sum += target_flow_stats['pair_count']
            target_flow_fast_pair_sum += target_flow_stats.get('fast_pair_count', 0)
            fine_target_flow_loss_sum += float(
                fine_target_flow_loss.detach().item()
            )
            fine_target_flow_pair_sum += fine_target_flow_stats['pair_count']
            trajectory_flow_loss_sum += float(trajectory_flow_loss.detach().item())
            trajectory_flow_pair_sum += trajectory_flow_stats['pair_count']
            target_center_loss_sum += float(target_center_loss.detach().item())
            target_level_center_loss_sum += float(
                target_level_center_loss.detach().item()
            )
            target_level_presence_loss_sum += float(
                target_level_presence_loss.detach().item()
            )
            target_level_velocity_loss_sum += float(
                target_level_velocity_loss.detach().item()
            )
            target_level_velocity_pair_sum += target_level_velocity_stats['pair_count']
            batch_count += 1
            pbar.set_postfix(
                loss='{:.5f}'.format(loss_sum / batch_count),
                pos='{:.4f}'.format(positive_fraction_sum / batch_count),
                pos_w='{:.2f}'.format(positive_weight_sum / batch_count),
                traj='{:.5f}'.format(trajectory_loss_sum / batch_count),
                conf='{:.5f}'.format(confidence_loss_sum / batch_count),
                metric_pd='{:.5f}'.format(metric_target_loss_sum / batch_count),
                metric_fa='{:.5f}'.format(metric_component_loss_sum / batch_count),
                hard_neg='{:.5f}'.format(hard_negative_loss_sum / batch_count),
                hard_frac='{:.4f}'.format(
                    hard_negative_fraction_sum / batch_count
                ),
                adv='{:.5f}'.format(advection_loss_sum / batch_count),
                flow_t='{:.5f}'.format(target_flow_loss_sum / batch_count),
                flow_n='{:d}'.format(target_flow_pair_sum),
                flow_fast='{:d}'.format(target_flow_fast_pair_sum),
                fine_flow='{:.5f}'.format(fine_target_flow_loss_sum / batch_count),
                fine_n='{:d}'.format(fine_target_flow_pair_sum),
                flow_2='{:.5f}'.format(trajectory_flow_loss_sum / batch_count),
                flow_2n='{:d}'.format(trajectory_flow_pair_sum),
                center='{:.5f}'.format(target_center_loss_sum / batch_count),
                tl_center='{:.5f}'.format(
                    target_level_center_loss_sum / batch_count
                ),
                tl_pres='{:.5f}'.format(
                    target_level_presence_loss_sum / batch_count
                ),
                tl_vel='{:.5f}'.format(
                    target_level_velocity_loss_sum / batch_count
                ),
                tl_n='{:d}'.format(target_level_velocity_pair_sum),
            )
        pbar.close()
        scheduler.step()

        print(
            'epoch {} motion sampling totals: fast_bin_count={}, selected_views={}'.format(
                epoch,
                dataset.fast_bin_count,
                dataset.motion_selected_views,
            )
        )
        print(
            'epoch {} trajectory augmentation totals: selected_views={}'.format(
                epoch,
                dataset.trajectory_augmented_views,
            )
        )
        print(
            'epoch {} M93 cross-video copy-paste totals: selected_views={}'.format(
                epoch,
                dataset.cross_video_copy_paste_views,
            )
        )
        print(
            'epoch {} horizontal flip augmentation totals: selected_views={}'.format(
                epoch,
                dataset.horizontal_flip_views,
            )
        )

        epoch_loss = loss_sum / max(batch_count, 1)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': epoch_loss,
            'temporal_memory': {
                'temporal_bin_size': int(cfg.temporal_memory_bin_size),
                'context_bins': int(cfg.temporal_memory_context_bins),
                'width': int(cfg.temporal_memory_width),
                'sequence_length': int(cfg.temporal_memory_sequence_length),
                'log_count_clip': float(cfg.temporal_memory_log_count_clip),
                'density_calibration_enabled': bool(
                    getattr(
                        cfg,
                        'temporal_frame_density_calibration_enabled',
                        False,
                    )
                ),
                'trajectory_extrapolation_enabled': trajectory_enabled,
                'confidence_head_enabled': confidence_head_enabled,
                'confidence_only_enabled': confidence_only_enabled,
                'advection_flow_only_enabled': advection_flow_only_enabled,
                'memory_only_enabled': memory_only_enabled,
                'local_temporal_context_enabled': local_temporal_context_enabled,
                'local_temporal_context_kernel_size': (
                    local_temporal_context_kernel_size
                ),
                'local_temporal_context_only_enabled': (
                    local_temporal_context_only_enabled
                ),
                'fine_memory_only_enabled': fine_memory_only_enabled,
                'target_center_enabled': target_center_enabled,
                'target_center_loss_weight': center_memory_loss_weight,
                'target_center_sigma': center_memory_sigma,
                'target_center_radius': center_memory_radius,
                'target_center_positive_loss_mass': center_memory_positive_loss_mass,
                'target_center_max_positive_weight': center_memory_max_positive_weight,
                'target_center_empty_loss_weight': center_memory_empty_loss_weight,
                'target_level_enabled': target_level_enabled,
                'target_level_center_loss_weight': target_level_center_loss_weight,
                'target_level_presence_loss_weight': target_level_presence_loss_weight,
                'target_level_velocity_loss_weight': target_level_velocity_loss_weight,
                'target_level_center_sigma': target_level_center_sigma,
                'target_level_center_radius': target_level_center_radius,
                'target_level_positive_loss_mass': target_level_positive_loss_mass,
                'target_level_max_positive_weight': target_level_max_positive_weight,
                'target_level_empty_loss_weight': target_level_empty_loss_weight,
                'target_level_velocity_huber_delta': target_level_velocity_huber_delta,
                'target_level_downsample': target_level_downsample,
                'center_memory_enabled': center_memory_enabled,
                'center_memory_channels': center_memory_channels,
                'center_memory_downsample': center_memory_downsample,
                'center_memory_only_enabled': center_memory_only_enabled,
                'temporal_attention_enabled': temporal_attention_enabled,
                'attention_output_init_std': temporal_attention_output_init_std,
                'attention_relative_bias_enabled': (
                    temporal_attention_relative_bias_enabled
                ),
                'attention_relative_bias_max_distance': (
                    temporal_attention_relative_bias_max_distance
                ),
                'advection_alignment_enabled': advection_alignment_enabled,
                'advection_alignment_loss_weight': advection_alignment_loss_weight,
                'advection_max_flow': advection_max_flow,
                'advection_target_flow_enabled': advection_target_flow_enabled,
                'advection_target_flow_weight': advection_target_flow_weight,
                'advection_target_flow_huber_delta': advection_target_flow_huber_delta,
                'advection_fast_motion_threshold': advection_fast_motion_threshold,
                'advection_fast_motion_weight': advection_fast_motion_weight,
                'advection_trajectory_flow_enabled': advection_trajectory_flow_enabled,
                'advection_trajectory_flow_weight': advection_trajectory_flow_weight,
                'advection_trajectory_flow_max_hop': advection_trajectory_flow_max_hop,
                'fine_temporal_memory_enabled': fine_temporal_memory_enabled,
                'fine_advection_max_flow': fine_advection_max_flow,
                'fine_target_flow_enabled': fine_target_flow_enabled,
                'fine_target_flow_weight': fine_target_flow_weight,
                'dense_only_enabled': dense_only_enabled,
                'dense_only_event_count_cutoff': dense_only_event_count_cutoff,
                'low_density_only_enabled': low_density_only_enabled,
                'low_density_only_event_count_cutoff': (
                    low_density_only_event_count_cutoff
                ),
                'max_videos_per_epoch': max_videos_per_epoch,
                'mid_density_sampling_enabled': mid_density_sampling_enabled,
                'mid_density_min_event_count': mid_density_min_event_count,
                'mid_density_max_event_count': mid_density_max_event_count,
                'mid_density_view_multiplier': mid_density_view_multiplier,
                'hard_negative_enabled': hard_negative_enabled,
                'hard_negative_weight': hard_negative_weight,
                'hard_negative_score_floor': hard_negative_score_floor,
                'motion_sampling_enabled': motion_sampling_enabled,
                'motion_sampling_min_event_count': motion_sampling_min_event_count,
                'motion_sampling_min_displacement': motion_sampling_min_displacement,
                'motion_sampling_probability': motion_sampling_probability,
                'motion_sampling_extra_views_only': motion_sampling_extra_views_only,
                'motion_sampling_fast_bin_count': dataset.fast_bin_count,
                'motion_sampling_selected_views': dataset.motion_selected_views,
                'trajectory_augmentation_enabled': (
                    dataset.trajectory_augmentation_enabled
                ),
                'trajectory_augmentation_selected_views': (
                    dataset.trajectory_augmented_views
                ),
                'cross_video_copy_paste_enabled': (
                    dataset.cross_video_copy_paste_enabled
                ),
                'cross_video_copy_paste_min_event_count': (
                    dataset.cross_video_copy_paste_min_event_count
                ),
                'cross_video_copy_paste_probability': (
                    dataset.cross_video_copy_paste_probability
                ),
                'cross_video_copy_paste_extra_views_only': (
                    dataset.cross_video_copy_paste_extra_views_only
                ),
                'cross_video_copy_paste_extra_views': (
                    dataset.cross_video_copy_paste_extra_views
                ),
                'cross_video_copy_paste_min_track_bins': (
                    dataset.cross_video_copy_paste_min_track_bins
                ),
                'cross_video_copy_paste_collision_radius': (
                    dataset.cross_video_copy_paste_collision_radius
                ),
                'cross_video_copy_paste_selected_views': (
                    dataset.cross_video_copy_paste_views
                ),
                'horizontal_flip_augmentation_enabled': (
                    horizontal_flip_augmentation_enabled
                ),
                'horizontal_flip_augmentation_probability': (
                    horizontal_flip_augmentation_probability
                ),
                'horizontal_flip_augmentation_selected_views': (
                    dataset.horizontal_flip_views
                ),
            },
        }
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            save_checkpoint(
                checkpoint,
                run_dir / 'best_loss_seed{}.pt'.format(cfg.seed),
            )
        save_checkpoint(
            checkpoint, run_dir / 'last_seed{}.pt'.format(cfg.seed)
        )
        if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint,
                run_dir / 'epoch_{:03d}_seed{}.pt'.format(epoch + 1, cfg.seed),
            )
        learning_rates = ', '.join(
            'lr_{}={:.8f}'.format(group['name'], group['lr'])
            for group in optimizer.param_groups
        )
        print(
            'epoch {}: loss={:.6f}, {}, best_loss={:.6f}'.format(
                epoch,
                epoch_loss,
                learning_rates,
                best_loss,
            )
        )

    summary = {
        'started_at': started_at.isoformat(timespec='seconds'),
        'seed': int(cfg.seed),
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'best_loss_checkpoint': str(
            run_dir / 'best_loss_seed{}.pt'.format(cfg.seed)
        ),
        'last_checkpoint': str(run_dir / 'last_seed{}.pt'.format(cfg.seed)),
        'config_overrides': list(cfg.config_overrides),
    }
    with (run_dir / 'run_summary.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
    print('best loss checkpoint:', summary['best_loss_checkpoint'])
    print('last checkpoint:', summary['last_checkpoint'])
