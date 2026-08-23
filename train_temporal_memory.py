"""Train a bidirectional full-stream temporal-memory event segmentation model."""

import json
import copy
import hashlib
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    teacher_selective_metric_loss,
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
from utils.target_frame_balanced import target_frame_balanced_positive_loss


def setup_seed(seed, strict_deterministic=False):
    seed = int(seed)
    if strict_deterministic:
        # M134 compares two separately launched CUDA jobs.  cuBLAS workspace
        # selection and TF32 can otherwise leave tiny, irreproducible updates
        # even when the random streams and cuDNN mode match.  PyTorch 1.9
        # still has unsupported deterministic backwards for grid sampling and
        # bilinear upsampling, so the global error-mode switch stays off.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        os.environ['EVSOD_DETERMINISTIC_WARP_CPU'] = '1'
        if hasattr(torch.backends, 'cuda'):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.allow_tf32 = False
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


def sha256_file(path):
    """Return a stable digest for checkpoint and fold-manifest provenance."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stable_state_digest(value, float_quantum=None):
    """Hash nested training state without relying on torch.save metadata."""
    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            if (
                float_quantum is not None
                and (tensor.is_floating_point() or tensor.is_complex())
            ):
                tensor = torch.round(
                    tensor.to(dtype=torch.float64) / float(float_quantum)
                ).to(dtype=torch.int64)
                digest.update(b'quantized_tensor\0')
            else:
                digest.update(b'tensor\0')
            digest.update(str(tensor.dtype).encode('utf-8'))
            digest.update(b'\0')
            digest.update(repr(tuple(tensor.shape)).encode('ascii'))
            digest.update(b'\0')
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b'ndarray\0')
            digest.update(str(array.dtype).encode('utf-8'))
            digest.update(b'\0')
            digest.update(repr(tuple(array.shape)).encode('ascii'))
            digest.update(b'\0')
            digest.update(array.tobytes())
            return
        if isinstance(item, dict):
            digest.update(b'dict\0')
            for key in sorted(item, key=lambda value: repr(value)):
                update(key)
                update(item[key])
            digest.update(b'\1')
            return
        if isinstance(item, (list, tuple)):
            digest.update(('tuple\0' if isinstance(item, tuple) else 'list\0').encode())
            for child in item:
                update(child)
            digest.update(b'\1')
            return
        if isinstance(item, (bool, int, float, str)) or item is None:
            digest.update(repr(item).encode('utf-8'))
            digest.update(b'\0')
            return
        digest.update(repr(item).encode('utf-8'))
        digest.update(b'\0')

    update(value)
    return digest.hexdigest()


def cpu_state_copy(value):
    """Detach nested optimizer/scheduler state for an audit sidecar."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state_copy(child) for key, child in value.items()}
    if isinstance(value, list):
        return [cpu_state_copy(child) for child in value]
    if isinstance(value, tuple):
        return tuple(cpu_state_copy(child) for child in value)
    return value


def training_state_audit(model, optimizer, scheduler):
    """Return stable hashes for the paired-control identity audit."""
    # Separate CUDA processes on the supported PyTorch 1.9 stack can differ
    # by a few ulps in convolution/upsample reductions. A 1e-7 bucket is
    # still crossed by values sitting on a bucket boundary despite a <1e-8
    # numerical difference, so use a 1e-6 identity bucket while retaining the
    # exact hashes for forensic reporting.
    pair_float_quantum = 1.0e-6
    return {
        'model_sha256': stable_state_digest(model.state_dict()),
        'optimizer_sha256': stable_state_digest(optimizer.state_dict()),
        'scheduler_sha256': stable_state_digest(scheduler.state_dict()),
        # PyTorch 1.9 still has a few CUDA reductions whose separate process
        # results differ by a handful of ulps.  Keep exact hashes for forensic
        # reporting and a documented quantized identity for the pair gate.
        'pair_float_quantum': pair_float_quantum,
        'model_quantized_sha256': stable_state_digest(
            model.state_dict(), float_quantum=pair_float_quantum
        ),
        'optimizer_quantized_sha256': stable_state_digest(
            optimizer.state_dict(), float_quantum=pair_float_quantum
        ),
        'scheduler_quantized_sha256': stable_state_digest(
            scheduler.state_dict(), float_quantum=pair_float_quantum
        ),
        'python_rng_sha256': stable_state_digest(random.getstate()),
        'numpy_rng_sha256': stable_state_digest(np.random.get_state()),
        'cpu_rng_sha256': stable_state_digest(torch.get_rng_state()),
        'cuda_rng_sha256': stable_state_digest(torch.cuda.get_rng_state_all()),
    }


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'cpu': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])


def copy_teacher_from_parent(model, parent_state):
    """Clone a clean frozen teacher after student forwards have started.

    The temporal model keeps autograd-connected diagnostics in private
    ``_last_*`` attributes.  PyTorch 1.9 cannot deepcopy those non-leaf
    tensors, so temporarily clear only the diagnostics while copying the
    module and then restore them on the student.
    """
    transient_names = (
        '_last_advection_consistency_loss',
        '_last_advection_forward_flows',
        '_last_fine_advection_forward_flows',
    )
    transient = {
        name: getattr(model, name, None) for name in transient_names
    }
    for name in transient_names:
        setattr(model, name, None)
    try:
        teacher = copy.deepcopy(model)
    finally:
        for name, value in transient.items():
            setattr(model, name, value)
    teacher.load_state_dict(parent_state, strict=True)
    return teacher


def build_scheduler(optimizer, config):
    scheduler_name = str(config.scheduler).lower()
    if scheduler_name == 'cosine':
        configured_t_max = getattr(config, 'scheduler_t_max', None)
        t_max = (
            int(config.epochs)
            if configured_t_max is None
            else int(configured_t_max)
        )
        if t_max <= 0:
            raise ValueError('TRAIN.scheduler_t_max must be positive when set.')
        if t_max < int(config.epochs):
            raise ValueError(
                'TRAIN.scheduler_t_max must be at least TRAIN.epochs.'
            )
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
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
    objectness_gate_enabled=False,
    objectness_gate_strength=0.5,
    objectness_gate_downsample=4,
    center_memory_enabled=False,
    local_temporal_context_enabled=False,
    normalization_max_groups=8,
    temporal_attention_num_heads=4,
):
    checkpoint_path = Path(str(checkpoint_path).strip())
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            'P23 initialization checkpoint not found: {}'.format(checkpoint_path)
        )
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    normalization_max_groups = int(normalization_max_groups)
    temporal_attention_num_heads = int(temporal_attention_num_heads)
    saved_memory = checkpoint.get('temporal_memory')
    if saved_memory is not None:
        saved_context_bins = saved_memory.get('context_bins')
        saved_width = saved_memory.get('width')
        saved_normalization_max_groups = int(
            saved_memory.get('normalization_max_groups', 8)
        )
        saved_temporal_attention_num_heads = int(
            saved_memory.get('temporal_attention_num_heads', 4)
        )
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
        if saved_normalization_max_groups != normalization_max_groups:
            raise ValueError(
                'M5 normalization_max_groups={} does not match {}.'.format(
                    saved_normalization_max_groups,
                    normalization_max_groups,
                )
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
        saved_objectness_gate = bool(
            saved_memory.get('objectness_gate_enabled', False)
        )
        saved_objectness_gate_strength = float(
            saved_memory.get('objectness_gate_strength', 0.5)
        )
        saved_objectness_gate_downsample = int(
            saved_memory.get('objectness_gate_downsample', 4)
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
        adding_objectness_gate = (
            bool(objectness_gate_enabled) and not saved_objectness_gate
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
        if saved_objectness_gate and not bool(objectness_gate_enabled):
            raise ValueError(
                'M5 objectness gate={} does not match configured {}.'.format(
                    saved_objectness_gate, objectness_gate_enabled
                )
            )
        if saved_objectness_gate and (
            saved_objectness_gate_strength != float(objectness_gate_strength)
            or saved_objectness_gate_downsample != int(objectness_gate_downsample)
        ):
            raise ValueError(
                'Objectness-gate metadata does not match the configured architecture.'
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
        if (
            saved_temporal_attention
            and saved_temporal_attention_num_heads != temporal_attention_num_heads
        ):
            raise ValueError(
                'Temporal-attention head count metadata does not match the '
                'configured architecture.'
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
                or adding_objectness_gate
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
            or adding_objectness_gate
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
            if adding_objectness_gate:
                for module_name in (
                    'objectness_center_head',
                    'objectness_presence_head',
                    'objectness_velocity_head',
                    'objectness_event_gate',
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
    objectness_flow_only_enabled=False,
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
    objectness_multiplier = float(
        getattr(config, 'temporal_memory_objectness_lr_multiplier', 1.0)
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
        or objectness_multiplier <= 0.0
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
    objectness_parameters = []
    objectness_auxiliary_parameters = []
    if getattr(model, 'objectness_gate_enabled', False):
        for module in (
            model.objectness_center_head,
            model.objectness_presence_head,
            model.objectness_velocity_head,
        ):
            objectness_auxiliary_parameters += list(module.parameters())
        for module in (
            model.objectness_center_head,
            model.objectness_presence_head,
            model.objectness_velocity_head,
            model.objectness_event_gate,
        ):
            objectness_parameters += list(module.parameters())
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
    if objectness_flow_only_enabled:
        if not getattr(model, 'advection_alignment_enabled', False):
            raise ValueError(
                'M132 objectness-flow-only mode requires advection alignment.'
            )
        if not objectness_auxiliary_parameters:
            raise ValueError(
                'M132 objectness-flow-only mode requires objectness heads.'
            )
        return optim.AdamW(
            [
                {
                    'name': 'advection_flow',
                    'params': list(model.flow_head.parameters()),
                    'lr': float(config.lr) * advection_multiplier,
                },
                {
                    'name': 'objectness_auxiliary',
                    'params': objectness_auxiliary_parameters,
                    'lr': float(config.lr) * objectness_multiplier,
                },
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
    if objectness_parameters:
        parameter_groups.append(
            {
                'name': 'objectness',
                'params': objectness_parameters,
                'lr': float(config.lr) * objectness_multiplier,
            }
        )
    return optim.AdamW(parameter_groups, weight_decay=1e-4)


def memory_config_summary(config):
    return (
        'enabled (bin_size={}, context_bins={}, width={}, norm_groups={}, '
        'attention_heads={}, sequence_length={}, '
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
        getattr(config, 'temporal_memory_normalization_max_groups', 8),
        getattr(config, 'temporal_memory_attention_num_heads', 4),
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

    pair_audit_requested = bool(
        str(
            getattr(
                cfg,
                'temporal_memory_teacher_selective_teacher_path',
                '',
            )
            or ''
        ).strip()
    )
    setup_seed(cfg.seed, strict_deterministic=pair_audit_requested)
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
        fold_manifest_path=getattr(
            cfg, 'temporal_memory_fold_manifest', ''
        ),
        train_folds=getattr(cfg, 'temporal_memory_train_folds', ''),
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
    target_frame_balanced_enabled = bool(
        getattr(cfg, 'temporal_memory_target_frame_balanced_enabled', False)
    )
    target_frame_balanced_weight = float(
        getattr(cfg, 'temporal_memory_target_frame_balanced_weight', 0.01)
    )
    target_frame_balanced_warmup_epochs = int(
        getattr(
            cfg,
            'temporal_memory_target_frame_balanced_warmup_epochs',
            0,
        )
    )
    target_frame_balanced_temporal_bin_size = int(
        getattr(
            cfg,
            'temporal_memory_target_frame_balanced_temporal_bin_size',
            cfg.temporal_memory_bin_size,
        )
    )
    teacher_selective_enabled = bool(
        getattr(
            cfg,
            'temporal_memory_teacher_selective_metric_enabled',
            False,
        )
    )
    teacher_selective_teacher_path = str(
        getattr(
            cfg,
            'temporal_memory_teacher_selective_teacher_path',
            '',
        )
        or ''
    ).strip()
    teacher_selective_target_weight = float(
        getattr(
            cfg,
            'temporal_memory_teacher_selective_target_weight',
            0.005,
        )
    )
    teacher_selective_component_weight = float(
        getattr(
            cfg,
            'temporal_memory_teacher_selective_component_weight',
            0.001,
        )
    )
    teacher_selective_warmup_epochs = int(
        getattr(
            cfg,
            'temporal_memory_teacher_selective_warmup_epochs',
            1,
        )
    )
    teacher_selective_threshold = float(
        getattr(
            cfg,
            'temporal_memory_teacher_selective_threshold',
            0.7226,
        )
    )
    teacher_selective_spatial_cell_size = int(
        getattr(
            cfg,
            'temporal_memory_teacher_selective_spatial_cell_size',
            3,
        )
    )
    teacher_selective_min_cell_events = int(
        getattr(
            cfg,
            'temporal_memory_teacher_selective_min_cell_events',
            2,
        )
    )
    teacher_selective_component_ratio = float(
        getattr(
            cfg,
            'temporal_memory_teacher_selective_component_ratio',
            0.01,
        )
    )
    m134_resume_checkpoint_path = str(
        getattr(
            cfg,
            'temporal_memory_m134_resume_checkpoint_path',
            '',
        )
        or ''
    ).strip()
    m134_resume_state_path = str(
        getattr(
            cfg,
            'temporal_memory_m134_resume_state_path',
            '',
        )
        or ''
    ).strip()
    m134_resume_start_epoch = int(
        getattr(
            cfg,
            'temporal_memory_m134_resume_start_epoch',
            0,
        )
    )
    m134_resume_enabled = bool(
        m134_resume_checkpoint_path or m134_resume_state_path
    )
    pair_audit_enabled = bool(teacher_selective_teacher_path)
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
    if target_frame_balanced_enabled:
        if target_frame_balanced_weight <= 0.0:
            raise ValueError(
                'TEMPORAL_MEMORY.target_frame_balanced_weight must be positive.'
            )
        if target_frame_balanced_warmup_epochs < 0:
            raise ValueError(
                'TEMPORAL_MEMORY.target_frame_balanced_warmup_epochs must be non-negative.'
            )
        if target_frame_balanced_temporal_bin_size <= 0:
            raise ValueError(
                'TEMPORAL_MEMORY.target_frame_balanced_temporal_bin_size must be positive.'
            )
        if target_frame_balanced_temporal_bin_size != int(
            cfg.temporal_memory_bin_size
        ):
            raise ValueError(
                'M137 target-frame bins must match TEMPORAL_MEMORY.bin_size.'
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
    objectness_gate_enabled = bool(
        getattr(cfg, 'temporal_memory_objectness_gate_enabled', False)
    )
    objectness_flow_only_enabled = bool(
        getattr(cfg, 'temporal_memory_objectness_flow_only_enabled', False)
    )
    objectness_gate_strength = float(
        getattr(cfg, 'temporal_memory_objectness_gate_strength', 0.50)
    )
    objectness_gate_downsample = int(
        getattr(cfg, 'temporal_memory_objectness_gate_downsample', 4)
    )
    objectness_center_loss_weight = float(
        getattr(cfg, 'temporal_memory_objectness_center_loss_weight', 0.025)
    )
    objectness_presence_loss_weight = float(
        getattr(cfg, 'temporal_memory_objectness_presence_loss_weight', 0.010)
    )
    objectness_velocity_loss_weight = float(
        getattr(cfg, 'temporal_memory_objectness_velocity_loss_weight', 0.010)
    )
    objectness_teacher_weight = float(
        getattr(cfg, 'temporal_memory_objectness_teacher_weight', 0.05)
    )
    objectness_preserve_weight = float(
        getattr(cfg, 'temporal_memory_objectness_preserve_weight', 0.50)
    )
    objectness_teacher_margin = float(
        getattr(cfg, 'temporal_memory_objectness_teacher_margin', 0.25)
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
        bool(objectness_flow_only_enabled),
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
    normalization_max_groups = int(
        getattr(cfg, 'temporal_memory_normalization_max_groups', 8)
    )
    temporal_attention_num_heads = int(
        getattr(cfg, 'temporal_memory_attention_num_heads', 4)
    )
    if normalization_max_groups <= 0:
        raise ValueError(
            'TEMPORAL_MEMORY.normalization_max_groups must be positive.'
        )
    if temporal_attention_num_heads <= 0:
        raise ValueError(
            'TEMPORAL_MEMORY.attention_num_heads must be positive.'
        )
    if (
        int(cfg.temporal_memory_width) * 6
    ) % temporal_attention_num_heads != 0:
        raise ValueError(
            'TEMPORAL_MEMORY.width * 6 must divide evenly into attention heads.'
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
    if objectness_gate_enabled and (
        target_center_enabled
        or confidence_head_enabled
        or target_level_enabled
        or center_memory_enabled
        or confidence_only_enabled
        or fine_memory_only_enabled
        or center_memory_only_enabled
        or advection_flow_only_enabled
        or memory_only_enabled
        or local_temporal_context_only_enabled
    ):
        raise ValueError(
            'Objectness gating cannot be combined with the existing auxiliary '
            'output branches in the first probe.'
        )
    if objectness_flow_only_enabled and not objectness_gate_enabled:
        raise ValueError(
            'M132 objectness-flow-only mode requires '
            'temporal_memory_objectness_gate_enabled=true.'
        )
    if objectness_flow_only_enabled and not advection_alignment_enabled:
        raise ValueError(
            'M132 objectness-flow-only mode requires advection alignment.'
        )
    if objectness_flow_only_enabled and not advection_target_flow_enabled:
        raise ValueError(
            'M132 objectness-flow-only mode requires target-centroid flow '
            'supervision.'
        )
    if objectness_gate_enabled:
        if objectness_gate_strength <= 0.0 or objectness_gate_downsample <= 0:
            raise ValueError('Objectness gate geometry/strength is invalid.')
        if (
            objectness_center_loss_weight <= 0.0
            or objectness_presence_loss_weight <= 0.0
            or objectness_velocity_loss_weight <= 0.0
            or objectness_teacher_weight <= 0.0
            or objectness_preserve_weight <= 0.0
        ):
            raise ValueError('Objectness loss weights must be positive.')
        if objectness_teacher_margin < 0.0:
            raise ValueError('Objectness teacher margin must be non-negative.')
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
    if pair_audit_enabled and not Path(teacher_selective_teacher_path).is_file():
        raise FileNotFoundError(
            'M134 teacher checkpoint not found: {}'.format(
                teacher_selective_teacher_path
            )
        )
    if pair_audit_enabled:
        if not teacher_selective_teacher_path:
            raise ValueError(
                'M134 pair-audit mode requires a teacher checkpoint path.'
            )
        if teacher_selective_target_weight <= 0.0:
            raise ValueError('M134 target loss weight must be positive.')
        if teacher_selective_component_weight <= 0.0:
            raise ValueError('M134 background loss weight must be positive.')
        if teacher_selective_warmup_epochs < 1:
            raise ValueError('M134 warmup must leave epoch 0 teacher-free.')
        if not 0.0 < teacher_selective_threshold < 1.0:
            raise ValueError('M134 threshold must be in (0, 1).')
        if teacher_selective_spatial_cell_size <= 0:
            raise ValueError('M134 spatial cell size must be positive.')
        if teacher_selective_min_cell_events <= 0:
            raise ValueError('M134 minimum cell events must be positive.')
        if not 0.0 < teacher_selective_component_ratio <= 1.0:
            raise ValueError('M134 component ratio must be in (0, 1].')
        incompatible = (
            metric_aux_enabled,
            trajectory_enabled,
            hard_negative_enabled,
            motion_sampling_enabled,
            bool(getattr(dataset, 'trajectory_augmentation_enabled', False)),
            bool(getattr(dataset, 'cross_video_copy_paste_enabled', False)),
            horizontal_flip_augmentation_enabled,
            int(getattr(cfg, 'temporal_memory_training_phase_offset', 0)) != 0,
            confidence_head_enabled,
            target_center_enabled,
            target_level_enabled,
            objectness_gate_enabled,
            fine_temporal_memory_enabled,
            local_temporal_context_enabled,
            confidence_only_enabled,
            fine_memory_only_enabled,
            center_memory_only_enabled,
            advection_flow_only_enabled,
            objectness_flow_only_enabled,
            memory_only_enabled,
            local_temporal_context_only_enabled,
        )
        if any(incompatible):
            raise ValueError(
                'M134 requires the plain M26 training path with all other '
                'auxiliary, augmentation, and isolated modes disabled.'
            )
    if m134_resume_enabled:
        if not (
            teacher_selective_enabled
            and pair_audit_enabled
            and m134_resume_checkpoint_path
            and m134_resume_state_path
        ):
            raise ValueError(
                'M134 resumed training requires the enabled treatment arm, '
                'its teacher path, and both anchor checkpoint/state paths.'
            )
        if m134_resume_start_epoch != teacher_selective_warmup_epochs:
            raise ValueError(
                'M134 resume must begin exactly at the teacher warmup epoch.'
            )
        if m134_resume_start_epoch <= 0 or m134_resume_start_epoch >= int(cfg.epochs):
            raise ValueError(
                'M134 resume start epoch must be in [1, TRAIN.epochs).'
            )
        if not Path(m134_resume_checkpoint_path).is_file():
            raise FileNotFoundError(
                'M134 resume checkpoint not found: {}'.format(
                    m134_resume_checkpoint_path
                )
            )
        if not Path(m134_resume_state_path).is_file():
            raise FileNotFoundError(
                'M134 resume state sidecar not found: {}'.format(
                    m134_resume_state_path
                )
            )
    model = BidirectionalTemporalMemoryNet(
        input_channels=int(cfg.temporal_memory_context_bins) * 2,
        width=int(cfg.temporal_memory_width),
        normalization_max_groups=normalization_max_groups,
        density_calibration_enabled=density_calibration_enabled,
        confidence_head_enabled=confidence_head_enabled,
        temporal_attention_enabled=temporal_attention_enabled,
        temporal_attention_num_heads=temporal_attention_num_heads,
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
        objectness_gate_enabled=objectness_gate_enabled,
        objectness_gate_strength=objectness_gate_strength,
        objectness_gate_downsample=objectness_gate_downsample,
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
        objectness_gate_enabled=objectness_gate_enabled,
        objectness_gate_strength=objectness_gate_strength,
        objectness_gate_downsample=objectness_gate_downsample,
        center_memory_enabled=center_memory_enabled,
        local_temporal_context_enabled=local_temporal_context_enabled,
        normalization_max_groups=normalization_max_groups,
        temporal_attention_num_heads=temporal_attention_num_heads,
    )
    initialized_from_sha256 = sha256_file(initialized_from)
    fold_manifest_path = str(
        getattr(cfg, 'temporal_memory_fold_manifest', '') or ''
    ).strip()
    fold_manifest_sha256 = (
        sha256_file(fold_manifest_path)
        if fold_manifest_path and Path(fold_manifest_path).is_file()
        else ''
    )
    teacher_selective_model = None
    teacher_selective_initial_state = None
    if pair_audit_enabled:
        # Keep the parent snapshot on CPU so the treatment does not allocate a
        # second GPU model during epoch 0.  The GPU teacher is materialized
        # lazily at warmup and restored from this immutable M26 state.
        teacher_selective_initial_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    teacher_selective_teacher_sha256 = ''
    if pair_audit_enabled:
        teacher_selective_teacher_sha256 = sha256_file(
            teacher_selective_teacher_path
        )
        if teacher_selective_teacher_sha256 != initialized_from_sha256:
            raise ValueError(
                'M134 teacher and parent checkpoint must be byte-identical: '
                'teacher_sha256={} parent_sha256={}'.format(
                    teacher_selective_teacher_sha256,
                    initialized_from_sha256,
                )
            )
    if teacher_selective_enabled:
        # The GPU copy is intentionally deferred until the first warmup epoch;
        # constructing it before epoch 0 changes CUDA workspace selection and
        # breaks the paired identity audit even when no teacher forward runs.
        teacher_selective_model = None
    m134_resume_payload = None
    m134_resume_state_payload = None
    m134_resume_expected_audit = None
    m134_resume_provenance = {
        'enabled': False,
        'checkpoint_path': '',
        'checkpoint_sha256': '',
        'state_path': '',
        'state_sha256': '',
        'start_epoch': 0,
        'anchor_state_verified': False,
    }
    if m134_resume_enabled:
        resume_checkpoint_path = Path(m134_resume_checkpoint_path).resolve()
        resume_state_path = Path(m134_resume_state_path).resolve()
        m134_resume_payload = torch.load(resume_checkpoint_path, map_location='cpu')
        m134_resume_state_payload = torch.load(resume_state_path, map_location='cpu')
        if not isinstance(m134_resume_payload, dict) or not isinstance(
            m134_resume_payload.get('model_state_dict'), dict
        ):
            raise ValueError('M134 resume checkpoint has no model_state_dict.')
        if int(m134_resume_payload.get('epoch', -1)) != m134_resume_start_epoch - 1:
            raise ValueError('M134 resume checkpoint epoch does not match start epoch.')
        resume_metadata = m134_resume_payload.get('temporal_memory', {})
        if not isinstance(resume_metadata, dict):
            raise ValueError('M134 resume checkpoint lacks temporal metadata.')
        if bool(resume_metadata.get('teacher_selective_metric_enabled', True)):
            raise ValueError('M134 resume anchor must be a control checkpoint.')
        if str(resume_metadata.get('init_model_sha256', '')) != initialized_from_sha256:
            raise ValueError('M134 resume anchor parent SHA mismatch.')
        if str(resume_metadata.get('fold_manifest_sha256', '')) != fold_manifest_sha256:
            raise ValueError('M134 resume anchor fold manifest SHA mismatch.')
        if not isinstance(m134_resume_state_payload, dict) or (
            m134_resume_state_payload.get('schema') != 'ev-uav-m134-state-audit-v1'
        ):
            raise ValueError('M134 resume state sidecar schema is invalid.')
        if int(m134_resume_state_payload.get('epoch', -1)) != m134_resume_start_epoch:
            raise ValueError('M134 resume state sidecar epoch does not match start epoch.')
        model.load_state_dict(m134_resume_payload['model_state_dict'], strict=True)
        m134_resume_provenance.update(
            {
                'enabled': True,
                'checkpoint_path': str(resume_checkpoint_path),
                'checkpoint_sha256': sha256_file(resume_checkpoint_path),
                'state_path': str(resume_state_path),
                'state_sha256': sha256_file(resume_state_path),
                'start_epoch': int(m134_resume_start_epoch),
            }
        )
    if target_frame_balanced_enabled:
        print(
            'M137 target-frame-balanced BCE: enabled '
            '(weight={:.4f}, warmup_epochs={}, temporal_bin_size={})'.format(
                target_frame_balanced_weight,
                target_frame_balanced_warmup_epochs,
                target_frame_balanced_temporal_bin_size,
            )
        )
    teacher_model = None
    if objectness_gate_enabled:
        # Keep a frozen copy of the released M26 decision boundary.  The
        # objectness branch is zero-residual at this point, so this teacher is
        # an exact identity reference while the new branch learns.
        teacher_model = copy.deepcopy(model).to(device)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
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
    elif objectness_flow_only_enabled:
        objectness_flow_parameter_ids = {
            id(parameter) for parameter in model.flow_head.parameters()
        }
        for module in (
            model.objectness_center_head,
            model.objectness_presence_head,
            model.objectness_velocity_head,
        ):
            objectness_flow_parameter_ids.update(
                id(parameter) for parameter in module.parameters()
            )
        for parameter in model.parameters():
            parameter.requires_grad = (
                id(parameter) in objectness_flow_parameter_ids
            )
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
        objectness_flow_only_enabled=objectness_flow_only_enabled,
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
    if objectness_flow_only_enabled:
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
                'M132 optimizer must contain exactly flow_head and the three '
                'training-only objectness heads.'
            )
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
    if m134_resume_enabled:
        optimizer.load_state_dict(
            m134_resume_state_payload['optimizer_state_dict']
        )
        scheduler.load_state_dict(
            m134_resume_state_payload['scheduler_state_dict']
        )
        expected_audit_path = (
            Path(m134_resume_checkpoint_path).resolve().parent
            / 'state_audit_epoch_{:03d}.json'.format(m134_resume_start_epoch)
        )
        if not expected_audit_path.is_file():
            raise FileNotFoundError(
                'M134 resume anchor audit is missing: {}'.format(expected_audit_path)
            )
        m134_resume_expected_audit = json.loads(
            expected_audit_path.read_text(encoding='utf-8')
        )
        resumed_audit = training_state_audit(model, optimizer, scheduler)
        for key in (
            'model_sha256',
            'optimizer_sha256',
            'scheduler_sha256',
        ):
            if resumed_audit.get(key) != m134_resume_expected_audit.get(key):
                raise AssertionError(
                    'M134 resumed {} does not exactly match its control anchor.'.format(
                        key
                    )
                )
        m134_resume_provenance['anchor_audit_path'] = str(
            expected_audit_path.resolve()
        )
        m134_resume_provenance['restore_state_audit'] = {
            key: resumed_audit[key]
            for key in ('model_sha256', 'optimizer_sha256', 'scheduler_sha256')
        }

    print('random seed:{}'.format(cfg.seed))
    print('run directory:', run_dir)
    print('config overrides:', ', '.join(cfg.config_overrides) or '(none)')
    print('temporal-memory model:', memory_config_summary(cfg))
    print('training videos:', len(dataset.file_paths))
    print(
        'fold filter: manifest={}, train_folds={}'.format(
            getattr(cfg, 'temporal_memory_fold_manifest', '') or '(none)',
            getattr(cfg, 'temporal_memory_train_folds', '') or '(all)',
        )
    )
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
    if objectness_flow_only_enabled:
        print(
            'M132 objectness-flow-only mode: M26 base/memory/attention and '
            'the zero event gate frozen; only flow and training-only '
            'objectness heads update.'
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
    if objectness_gate_enabled:
        print(
            'M123 objectness gate: enabled '
            '(strength={:.3f}, downsample={}, teacher_weight={:.3f}, '
            'preserve_weight={:.3f})'.format(
                objectness_gate_strength,
                objectness_gate_downsample,
                objectness_teacher_weight,
                objectness_preserve_weight,
            )
        )
    if teacher_selective_enabled:
        print(
            'M134 teacher-selective metric: enabled '
            '(warmup_epochs={}, threshold={:.4f}, target_weight={:.4f}, '
            'background_weight={:.4f}, teacher_sha256={})'.format(
                teacher_selective_warmup_epochs,
                teacher_selective_threshold,
                teacher_selective_target_weight,
                teacher_selective_component_weight,
                teacher_selective_teacher_sha256,
            )
        )

    best_loss = float('inf')
    best_epoch = None
    teacher_selective_epoch_diagnostics = []
    target_frame_balanced_epoch_diagnostics = []
    for epoch in range(m134_resume_start_epoch, int(cfg.epochs)):
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
        elif objectness_flow_only_enabled:
            model.eval()
            model.flow_head.train()
            model.objectness_center_head.train()
            model.objectness_presence_head.train()
            model.objectness_velocity_head.train()
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
        if (
            teacher_selective_enabled
            and epoch >= teacher_selective_warmup_epochs
            and teacher_selective_model is None
        ):
            if teacher_selective_initial_state is None:
                raise RuntimeError('M134 teacher parent state was not captured.')
            # deepcopy does not initialize modules, but preserve every RNG
            # stream explicitly because this allocation occurs mid-training.
            rng_before_teacher_copy = capture_rng_state()
            teacher_selective_model = copy_teacher_from_parent(
                model,
                teacher_selective_initial_state,
            )
            restore_rng_state(rng_before_teacher_copy)
            teacher_selective_model.eval()
            for parameter in teacher_selective_model.parameters():
                parameter.requires_grad_(False)
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
        objectness_center_loss_sum = 0.0
        objectness_presence_loss_sum = 0.0
        objectness_velocity_loss_sum = 0.0
        objectness_teacher_loss_sum = 0.0
        objectness_preserve_loss_sum = 0.0
        objectness_velocity_pair_sum = 0
        teacher_selective_recall_loss_sum = 0.0
        teacher_selective_preserve_loss_sum = 0.0
        teacher_selective_background_loss_sum = 0.0
        teacher_selective_missed_group_sum = 0
        teacher_selective_covered_group_sum = 0
        teacher_selective_hard_bg_sum = 0
        teacher_selective_target_group_sum = 0
        teacher_selective_candidate_cell_sum = 0
        target_frame_balanced_loss_sum = 0.0
        target_frame_balanced_group_sum = 0
        batch_count = 0
        if m134_resume_enabled and epoch == m134_resume_start_epoch:
            # A single-worker DataLoader draws its base seed when its iterator
            # is created. The source control did that in epoch 0; creating the
            # resumed epoch-1 iterator here restores the same CPU RNG point
            # before the first treatment forward without fetching a batch.
            data_iterator = iter(dataloader)
            pre_treatment_audit = training_state_audit(model, optimizer, scheduler)
            for key in (
                'model_sha256',
                'optimizer_sha256',
                'scheduler_sha256',
                'python_rng_sha256',
                'numpy_rng_sha256',
                'cpu_rng_sha256',
                'cuda_rng_sha256',
            ):
                if pre_treatment_audit.get(key) != m134_resume_expected_audit.get(key):
                    raise AssertionError(
                        'M134 pre-treatment {} does not match control epoch 1.'.format(
                            key
                        )
                    )
            m134_resume_provenance['anchor_state_verified'] = True
            m134_resume_provenance['anchor_state_audit'] = {
                key: pre_treatment_audit[key]
                for key in (
                    'model_sha256',
                    'optimizer_sha256',
                    'scheduler_sha256',
                    'python_rng_sha256',
                    'numpy_rng_sha256',
                    'cpu_rng_sha256',
                    'cuda_rng_sha256',
                )
            }
            (run_dir / 'm134_resume_anchor_audit.json').write_text(
                json.dumps(m134_resume_provenance, indent=2, sort_keys=True),
                encoding='utf-8',
            )
            pbar_input = data_iterator
            pbar_total = len(dataloader)
        else:
            pbar_input = dataloader
            pbar_total = None
        pbar = tqdm.tqdm(
            pbar_input,
            total=pbar_total,
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
            event_times = batch['event_times'].to(device, non_blocking=True)
            event_y = batch['event_y'].to(device, non_blocking=True)
            event_x = batch['event_x'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            target_ids = batch['target_ids'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            model_output = model(
                frames,
                return_target_center_logits=target_center_enabled,
                return_target_level_outputs=target_level_enabled,
                return_objectness_outputs=objectness_gate_enabled,
            )
            objectness_center_logits = None
            objectness_presence_logits = None
            objectness_velocity_maps = None
            target_level_center_logits = None
            target_level_presence_logits = None
            target_level_velocity_maps = None
            if objectness_gate_enabled:
                (
                    logit_maps,
                    objectness_center_logits,
                    objectness_presence_logits,
                    objectness_velocity_maps,
                ) = model_output
                logit_maps = logit_maps.squeeze(0)
                objectness_center_logits = objectness_center_logits.squeeze(0)
                objectness_presence_logits = objectness_presence_logits.squeeze(0)
                objectness_velocity_maps = objectness_velocity_maps.squeeze(0)
                confidence_logit_maps = None
                target_center_logits = None
            elif target_level_enabled:
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
            target_frame_balanced_loss = event_logits.sum() * 0.0
            target_frame_balanced_group_count = 0
            if (
                target_frame_balanced_enabled
                and epoch >= target_frame_balanced_warmup_epochs
            ):
                target_frame_locations = torch.stack(
                    (
                        torch.zeros_like(event_x),
                        event_x,
                        event_y,
                        event_times,
                    ),
                    dim=1,
                )
                (
                    target_frame_balanced_loss,
                    target_frame_balanced_group_count,
                ) = target_frame_balanced_positive_loss(
                    event_logits,
                    labels,
                    target_ids,
                    target_frame_locations,
                    target_frame_balanced_temporal_bin_size,
                    from_logits=True,
                )
                loss = loss + (
                    target_frame_balanced_weight
                    * target_frame_balanced_loss
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
            objectness_teacher_loss = event_logits.sum() * 0.0
            objectness_preserve_loss = event_logits.sum() * 0.0
            if objectness_gate_enabled:
                with torch.no_grad():
                    teacher_logit_maps = teacher_model(frames).squeeze(0)
                    teacher_event_logits = teacher_logit_maps[
                        event_time_indices,
                        0,
                        event_y,
                        event_x,
                    ]
                objectness_teacher_loss = F.smooth_l1_loss(
                    event_logits,
                    teacher_event_logits,
                )
                preserve_mask = (labels > 0.5) | (
                    torch.sigmoid(teacher_event_logits) > 0.80
                )
                if bool(preserve_mask.any()):
                    objectness_preserve_loss = F.relu(
                        teacher_event_logits[preserve_mask]
                        - event_logits[preserve_mask]
                        - objectness_teacher_margin
                    ).pow(2).mean()
                loss = loss + (
                    objectness_teacher_weight * objectness_teacher_loss
                    + objectness_preserve_weight * objectness_preserve_loss
                )
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
            objectness_center_loss = event_logits.sum() * 0.0
            objectness_presence_loss = event_logits.sum() * 0.0
            objectness_velocity_loss = event_logits.sum() * 0.0
            objectness_velocity_stats = {'pair_count': 0, 'mean_motion': 0.0}
            if objectness_gate_enabled:
                objectness_height = objectness_center_logits.shape[2]
                objectness_width = objectness_center_logits.shape[3]
                objectness_x = torch.round(
                    event_x.float()
                    * float(objectness_width - 1)
                    / max(int(cfg.res[0]) - 1, 1)
                ).long()
                objectness_y = torch.round(
                    event_y.float()
                    * float(objectness_height - 1)
                    / max(int(cfg.res[1]) - 1, 1)
                ).long()
                objectness_scale = min(
                    float(objectness_width) / float(cfg.res[0]),
                    float(objectness_height) / float(cfg.res[1]),
                )
                objectness_heatmaps = build_target_center_heatmaps(
                    objectness_x,
                    objectness_y,
                    labels,
                    target_ids,
                    event_time_indices,
                    batch_size=objectness_center_logits.shape[0],
                    height=objectness_height,
                    width=objectness_width,
                    sigma=max(0.5, target_level_center_sigma * objectness_scale),
                    radius=max(
                        1,
                        int(round(target_level_center_radius * objectness_scale)),
                    ),
                )
                objectness_center_loss, _ = target_center_heatmap_loss(
                    objectness_center_logits,
                    objectness_heatmaps,
                    target_positive_loss_mass=target_level_positive_loss_mass,
                    max_positive_weight=target_level_max_positive_weight,
                    empty_loss_weight=target_level_empty_loss_weight,
                )
                objectness_presence_loss, _ = target_level_presence_loss_fn(
                    objectness_presence_logits,
                    event_time_indices,
                    labels,
                    target_ids,
                )
                objectness_velocity_loss, objectness_velocity_stats = (
                    target_level_velocity_loss_fn(
                        objectness_velocity_maps,
                        event_time_indices,
                        objectness_x,
                        objectness_y,
                        labels,
                        target_ids,
                        huber_delta=target_level_velocity_huber_delta,
                    )
                )
                loss = loss + (
                    objectness_center_loss_weight * objectness_center_loss
                    + objectness_presence_loss_weight * objectness_presence_loss
                    + objectness_velocity_loss_weight * objectness_velocity_loss
                )
            metric_target_loss = event_logits.sum() * 0.0
            metric_component_loss = event_logits.sum() * 0.0
            event_locations = None
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
            teacher_selective_target_loss = event_logits.sum() * 0.0
            teacher_selective_background_loss = event_logits.sum() * 0.0
            teacher_selective_stats = {
                'recall_loss': event_logits.sum() * 0.0,
                'preserve_loss': event_logits.sum() * 0.0,
                'target_group_count': 0,
                'missed_group_count': 0,
                'covered_group_count': 0,
                'candidate_cell_count': 0,
                'hard_bg_count': 0,
            }
            if teacher_selective_enabled and epoch >= teacher_selective_warmup_epochs:
                if event_locations is None:
                    event_locations = torch.stack(
                        (
                            torch.zeros_like(event_x),
                            event_x,
                            event_y,
                            event_time_indices
                            * int(cfg.temporal_memory_bin_size)
                            + 1,
                        ),
                        dim=1,
                    )
                if teacher_selective_model is None:
                    raise RuntimeError('M134 teacher model was not initialized.')
                with torch.no_grad():
                    teacher_output = teacher_selective_model(frames)
                    if not torch.is_tensor(teacher_output):
                        raise RuntimeError(
                            'M134 teacher must return plain event logits; '
                            'disable auxiliary model branches.'
                        )
                    teacher_logit_maps = teacher_output.squeeze(0)
                    teacher_event_logits = teacher_logit_maps[
                        event_time_indices,
                        0,
                        event_y,
                        event_x,
                    ]
                (
                    teacher_selective_target_loss,
                    teacher_selective_background_loss,
                    teacher_selective_stats,
                ) = teacher_selective_metric_loss(
                    event_logits,
                    teacher_event_logits,
                    labels,
                    target_ids,
                    event_locations,
                    int(cfg.temporal_memory_bin_size),
                    teacher_selective_threshold,
                    teacher_selective_spatial_cell_size,
                    teacher_selective_min_cell_events,
                    teacher_selective_component_ratio,
                )
                loss = loss + (
                    teacher_selective_target_weight
                    * teacher_selective_target_loss
                    + teacher_selective_component_weight
                    * teacher_selective_background_loss
                )
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
            objectness_center_loss_sum += float(
                objectness_center_loss.detach().item()
            )
            objectness_presence_loss_sum += float(
                objectness_presence_loss.detach().item()
            )
            objectness_velocity_loss_sum += float(
                objectness_velocity_loss.detach().item()
            )
            objectness_teacher_loss_sum += float(
                objectness_teacher_loss.detach().item()
            )
            objectness_preserve_loss_sum += float(
                objectness_preserve_loss.detach().item()
            )
            objectness_velocity_pair_sum += objectness_velocity_stats['pair_count']
            teacher_selective_recall_loss_sum += float(
                teacher_selective_stats['recall_loss'].detach().item()
            )
            teacher_selective_preserve_loss_sum += float(
                teacher_selective_stats['preserve_loss'].detach().item()
            )
            teacher_selective_background_loss_sum += float(
                teacher_selective_background_loss.detach().item()
            )
            teacher_selective_target_group_sum += int(
                teacher_selective_stats['target_group_count']
            )
            teacher_selective_missed_group_sum += int(
                teacher_selective_stats['missed_group_count']
            )
            teacher_selective_covered_group_sum += int(
                teacher_selective_stats['covered_group_count']
            )
            teacher_selective_candidate_cell_sum += int(
                teacher_selective_stats['candidate_cell_count']
            )
            teacher_selective_hard_bg_sum += int(
                teacher_selective_stats['hard_bg_count']
            )
            target_frame_balanced_loss_sum += float(
                target_frame_balanced_loss.detach().item()
            )
            target_frame_balanced_group_sum += int(
                target_frame_balanced_group_count
            )
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
                obj_center='{:.5f}'.format(
                    objectness_center_loss_sum / batch_count
                ),
                obj_pres='{:.5f}'.format(
                    objectness_presence_loss_sum / batch_count
                ),
                obj_vel='{:.5f}'.format(
                    objectness_velocity_loss_sum / batch_count
                ),
                obj_teacher='{:.5f}'.format(
                    objectness_teacher_loss_sum / batch_count
                ),
                obj_keep='{:.5f}'.format(
                    objectness_preserve_loss_sum / batch_count
                ),
                obj_n='{:d}'.format(objectness_velocity_pair_sum),
                m134_rec='{:.5f}'.format(
                    teacher_selective_recall_loss_sum / batch_count
                ),
                m134_keep='{:.5f}'.format(
                    teacher_selective_preserve_loss_sum / batch_count
                ),
                m134_bg='{:.5f}'.format(
                    teacher_selective_background_loss_sum / batch_count
                ),
                m134_miss='{:d}'.format(teacher_selective_missed_group_sum),
                m134_cov='{:d}'.format(teacher_selective_covered_group_sum),
                m134_bg_n='{:d}'.format(teacher_selective_hard_bg_sum),
                m137_tf='{:.5f}'.format(
                    target_frame_balanced_loss_sum / batch_count
                ),
                m137_n='{:d}'.format(target_frame_balanced_group_sum),
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
        if target_frame_balanced_enabled:
            target_frame_balanced_epoch_diagnostics.append(
                {
                    'epoch': int(epoch + 1),
                    'zero_based_epoch': int(epoch),
                    'batch_count': int(batch_count),
                    'loss_mean': float(
                        target_frame_balanced_loss_sum / max(batch_count, 1)
                    ),
                    'target_group_count': int(target_frame_balanced_group_sum),
                    'target_groups_per_batch': float(
                        target_frame_balanced_group_sum / max(batch_count, 1)
                    ),
                }
            )
            (run_dir / 'm137_target_frame_balanced_diagnostics.json').write_text(
                json.dumps(
                    target_frame_balanced_epoch_diagnostics,
                    indent=2,
                    sort_keys=True,
                ),
                encoding='utf-8',
            )
        state_audit = None
        if pair_audit_enabled:
            state_audit = training_state_audit(model, optimizer, scheduler)
            teacher_selective_epoch_diagnostics.append(
                {
                    'epoch': int(epoch + 1),
                    'zero_based_epoch': int(epoch),
                    'batch_count': int(batch_count),
                    'loss_mean': float(epoch_loss),
                    'recall_loss_mean': float(
                        teacher_selective_recall_loss_sum / max(batch_count, 1)
                    ),
                    'preserve_loss_mean': float(
                        teacher_selective_preserve_loss_sum / max(batch_count, 1)
                    ),
                    'background_loss_mean': float(
                        teacher_selective_background_loss_sum / max(batch_count, 1)
                    ),
                    'target_group_count': int(teacher_selective_target_group_sum),
                    'missed_group_count': int(teacher_selective_missed_group_sum),
                    'covered_group_count': int(teacher_selective_covered_group_sum),
                    'candidate_cell_count': int(
                        teacher_selective_candidate_cell_sum
                    ),
                    'hard_bg_count': int(teacher_selective_hard_bg_sum),
                    'state_audit': state_audit,
                }
            )
            (run_dir / 'm134_epoch_diagnostics.json').write_text(
                json.dumps(teacher_selective_epoch_diagnostics, indent=2, sort_keys=True),
                encoding='utf-8',
            )
            (run_dir / 'state_audit_epoch_{:03d}.json'.format(epoch + 1)).write_text(
                json.dumps(
                    {
                        'epoch': int(epoch + 1),
                        'zero_based_epoch': int(epoch),
                        **state_audit,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding='utf-8',
            )
            if epoch == 0:
                # The deployment checkpoint intentionally omits optimizer and
                # RNG state. Keep one CPU sidecar so the pair audit can use a
                # numeric tolerance instead of a hash bucket at epoch 1.
                torch.save(
                    {
                        'schema': 'ev-uav-m134-state-audit-v1',
                        'epoch': int(epoch + 1),
                        'optimizer_state_dict': cpu_state_copy(
                            optimizer.state_dict()
                        ),
                        'scheduler_state_dict': cpu_state_copy(
                            scheduler.state_dict()
                        ),
                    },
                    run_dir / 'state_audit_epoch_{:03d}.pt'.format(epoch + 1),
                )
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': epoch_loss,
            'temporal_memory': {
                'temporal_bin_size': int(cfg.temporal_memory_bin_size),
                'context_bins': int(cfg.temporal_memory_context_bins),
                'width': int(cfg.temporal_memory_width),
                'normalization_max_groups': normalization_max_groups,
                'sequence_length': int(cfg.temporal_memory_sequence_length),
                'log_count_clip': float(cfg.temporal_memory_log_count_clip),
                'fold_manifest': str(
                    getattr(cfg, 'temporal_memory_fold_manifest', '')
                ),
                'fold_manifest_sha256': fold_manifest_sha256,
                'train_folds': str(
                    getattr(cfg, 'temporal_memory_train_folds', '')
                ),
                'selected_folds': (
                    list(dataset.selected_folds)
                    if dataset.selected_folds is not None else None
                ),
                'init_model_path': str(Path(initialized_from).resolve()),
                'init_model_sha256': initialized_from_sha256,
                'teacher_selective_metric_enabled': teacher_selective_enabled,
                'teacher_selective_teacher_path': (
                    str(Path(teacher_selective_teacher_path).resolve())
                    if teacher_selective_teacher_path else ''
                ),
                'teacher_selective_teacher_sha256': teacher_selective_teacher_sha256,
                'teacher_selective_target_weight': teacher_selective_target_weight,
                'teacher_selective_component_weight': (
                    teacher_selective_component_weight
                ),
                'teacher_selective_warmup_epochs': teacher_selective_warmup_epochs,
                'teacher_selective_threshold': teacher_selective_threshold,
                'teacher_selective_spatial_cell_size': (
                    teacher_selective_spatial_cell_size
                ),
                'teacher_selective_min_cell_events': (
                    teacher_selective_min_cell_events
                ),
                'teacher_selective_component_ratio': (
                    teacher_selective_component_ratio
                ),
                'm134_resume': m134_resume_provenance,
                'teacher_selective_state_audit': state_audit,
                'target_frame_balanced_enabled': target_frame_balanced_enabled,
                'target_frame_balanced_weight': target_frame_balanced_weight,
                'target_frame_balanced_warmup_epochs': (
                    target_frame_balanced_warmup_epochs
                ),
                'target_frame_balanced_temporal_bin_size': (
                    target_frame_balanced_temporal_bin_size
                ),
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
                'objectness_flow_only_enabled': objectness_flow_only_enabled,
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
                'objectness_gate_enabled': objectness_gate_enabled,
                'objectness_gate_strength': objectness_gate_strength,
                'objectness_gate_downsample': objectness_gate_downsample,
                'objectness_center_loss_weight': objectness_center_loss_weight,
                'objectness_presence_loss_weight': objectness_presence_loss_weight,
                'objectness_velocity_loss_weight': objectness_velocity_loss_weight,
                'objectness_teacher_weight': objectness_teacher_weight,
                'objectness_preserve_weight': objectness_preserve_weight,
                'objectness_teacher_margin': objectness_teacher_margin,
                'center_memory_enabled': center_memory_enabled,
                'center_memory_channels': center_memory_channels,
                'center_memory_downsample': center_memory_downsample,
                'center_memory_only_enabled': center_memory_only_enabled,
                'temporal_attention_enabled': temporal_attention_enabled,
                'temporal_attention_num_heads': temporal_attention_num_heads,
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
        'start_epoch': int(m134_resume_start_epoch),
        'm134_resume': m134_resume_provenance,
        'config_overrides': list(cfg.config_overrides),
    }
    with (run_dir / 'run_summary.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
    print('best loss checkpoint:', summary['best_loss_checkpoint'])
    print('last checkpoint:', summary['last_checkpoint'])
