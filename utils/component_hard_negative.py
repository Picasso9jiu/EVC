"""Training-only target-frame and false-component auxiliary losses."""

import math

import torch


def _zero_loss(predictions):
    return predictions.reshape(-1).sum() * 0


def _group_max(values, inverse, group_count):
    """Compute differentiable per-group maxima on the repository's torch version."""
    if group_count <= 0 or values.numel() == 0:
        return values.new_empty((0,))
    order = torch.argsort(inverse)
    sorted_inverse = inverse[order]
    sorted_values = values[order]
    starts = torch.cat(
        (
            inverse.new_zeros(1),
            torch.nonzero(
                sorted_inverse[1:] != sorted_inverse[:-1],
                as_tuple=False,
            ).reshape(-1) + 1,
        )
    )
    ends = torch.cat((starts[1:], starts.new_tensor([sorted_values.numel()])))
    peaks = [
        sorted_values[int(start):int(end)].max()
        for start, end in zip(starts.tolist(), ends.tolist())
    ]
    if len(peaks) != int(group_count):
        raise RuntimeError(
            'Grouped maxima produced {} groups; expected {}.'.format(
                len(peaks), group_count
            )
        )
    return torch.stack(peaks)


def _group_sum(values, inverse, group_count):
    """Compute per-group sums without CUDA atomic scatter operations."""
    if group_count <= 0 or values.numel() == 0:
        return values.new_empty((0,))
    order = torch.argsort(inverse)
    sorted_inverse = inverse[order]
    sorted_values = values[order]
    _, counts = torch.unique_consecutive(sorted_inverse, return_counts=True)
    if int(counts.numel()) != int(group_count):
        raise RuntimeError(
            'Grouped sums produced {} groups; expected {}.'.format(
                counts.numel(), group_count
            )
        )
    ends = torch.cumsum(counts, dim=0)
    starts = ends - counts
    prefix = torch.cat((sorted_values.new_zeros(1), torch.cumsum(sorted_values, dim=0)))
    return prefix.index_select(0, ends) - prefix.index_select(0, starts)


def teacher_selective_metric_loss(
    student_logits,
    teacher_logits,
    labels,
    target_ids,
    locations,
    temporal_bin_size,
    decision_threshold,
    spatial_cell_size,
    min_cell_events,
    component_ratio,
):
    """Apply the fixed M134 teacher-selective target and background losses.

    The teacher selects target windows and hard background cells.  It is detached
    here so only the student contributes gradients; grouped maxima match the
    official "at least one event in a target window" decision rule.
    """
    student_logits = student_logits.reshape(-1)
    teacher_logits = teacher_logits.reshape(-1).detach()
    labels = labels.reshape(-1)
    target_ids = target_ids.reshape(-1).long()
    if locations.ndim != 2 or locations.shape[1] < 4:
        raise ValueError('locations must have shape [N, >=4].')
    if not (
        student_logits.numel()
        == teacher_logits.numel()
        == labels.numel()
        == target_ids.numel()
        == locations.shape[0]
    ):
        raise ValueError(
            'student, teacher, label, target-id, and location counts must match.'
        )
    if temporal_bin_size <= 0:
        raise ValueError('temporal_bin_size must be positive.')
    if not 0.0 < float(decision_threshold) < 1.0:
        raise ValueError('decision_threshold must be in (0, 1).')
    if spatial_cell_size <= 0 or min_cell_events <= 0:
        raise ValueError('Background cell geometry must be positive.')
    if not 0.0 < float(component_ratio) <= 1.0:
        raise ValueError('component_ratio must be in (0, 1].')

    threshold_logit = math.log(
        float(decision_threshold) / (1.0 - float(decision_threshold))
    )
    zero = _zero_loss(student_logits)
    stats = {
        'target_group_count': 0,
        'missed_group_count': 0,
        'covered_group_count': 0,
        'candidate_cell_count': 0,
        'hard_bg_count': 0,
    }

    event_times = locations[:, 3].long()
    batch_ids = locations[:, 0].long()
    positive_mask = (labels > 0.5) & (target_ids > 0)
    recall_loss = zero
    preserve_loss = zero
    if torch.any(positive_mask):
        positive_target_ids = target_ids[positive_mask]
        positive_times = event_times[positive_mask]
        positive_batches = batch_ids[positive_mask]
        time_bins = torch.div(
            positive_times,
            int(temporal_bin_size),
            rounding_mode='floor',
        )
        target_stride = int(positive_target_ids.max().item()) + 1
        time_stride = int(time_bins.max().item()) + 1
        group_keys = (
            (positive_batches * target_stride + positive_target_ids)
            * time_stride
            + time_bins
        )
        _, inverse = torch.unique(group_keys, sorted=True, return_inverse=True)
        group_count = int(inverse.max().item()) + 1
        teacher_peaks = _group_max(
            teacher_logits[positive_mask], inverse, group_count
        )
        student_peaks = _group_max(
            student_logits[positive_mask], inverse, group_count
        )
        missed = teacher_peaks < float(threshold_logit)
        covered = ~missed
        if torch.any(missed):
            recall_loss = torch.relu(
                float(threshold_logit) - student_peaks[missed]
            ).pow(2).mean()
        if torch.any(covered):
            preserve_loss = torch.relu(
                teacher_peaks[covered] - student_peaks[covered]
            ).pow(2).mean()
        stats.update(
            {
                'target_group_count': group_count,
                'missed_group_count': int(missed.sum().item()),
                'covered_group_count': int(covered.sum().item()),
            }
        )

    # Select hard background cells using only detached teacher peaks and labels.
    background_loss = zero
    if student_logits.numel() > 0:
        coordinates = locations.long()
        cell_x = torch.div(
            coordinates[:, 1], int(spatial_cell_size), rounding_mode='floor'
        )
        cell_y = torch.div(
            coordinates[:, 2], int(spatial_cell_size), rounding_mode='floor'
        )
        cell_time = torch.div(
            coordinates[:, 3], int(temporal_bin_size), rounding_mode='floor'
        )
        x_stride = int(cell_x.max().item()) + 1
        y_stride = int(cell_y.max().item()) + 1
        time_stride = int(cell_time.max().item()) + 1
        cell_keys = (
            ((batch_ids * time_stride + cell_time) * y_stride + cell_y)
            * x_stride
            + cell_x
        )
        _, cell_inverse = torch.unique(
            cell_keys, sorted=True, return_inverse=True
        )
        cell_count = int(cell_inverse.max().item()) + 1
        event_counts = _group_sum(
            torch.ones_like(student_logits), cell_inverse, cell_count
        )
        positive_counts = _group_sum(
            (labels > 0.5).to(dtype=student_logits.dtype),
            cell_inverse,
            cell_count,
        )
        candidate_mask = (positive_counts == 0) & (
            event_counts >= int(min_cell_events)
        )
        candidate_indices = torch.nonzero(
            candidate_mask, as_tuple=False
        ).reshape(-1)
        candidate_count = int(candidate_indices.numel())
        stats['candidate_cell_count'] = candidate_count
        if candidate_count > 0:
            hard_count = max(
                1, int(math.ceil(candidate_count * float(component_ratio)))
            )
            teacher_cell_peaks = _group_max(
                teacher_logits, cell_inverse, cell_count
            ).detach()
            student_cell_peaks = _group_max(
                student_logits, cell_inverse, cell_count
            )
            top = torch.topk(
                teacher_cell_peaks[candidate_indices],
                k=min(hard_count, candidate_count),
                largest=True,
                sorted=False,
            ).indices
            hard_indices = candidate_indices[top]
            background_loss = torch.relu(
                student_cell_peaks[hard_indices] - float(threshold_logit)
            ).pow(2).mean()
            stats['hard_bg_count'] = int(hard_indices.numel())

    return recall_loss + preserve_loss, background_loss, {
        'recall_loss': recall_loss,
        'preserve_loss': preserve_loss,
        **stats,
    }


def _noisy_or(event_activation, inverse, group_count, eps):
    """Aggregate event activations into differentiable group activations."""
    log_not_activation = torch.log1p(
        -event_activation.clamp(max=1.0 - float(eps))
    )
    group_log_not_activation = event_activation.new_zeros(
        group_count
    ).scatter_add(0, inverse, log_not_activation)
    return 1.0 - torch.exp(group_log_not_activation)


def target_frame_activation_loss(
    predictions,
    labels,
    target_ids,
    locations,
    temporal_bin_size,
    activation_threshold,
    activation_temperature,
    eps=1e-5,
):
    """Encourage one confident event in every official target-time frame.

    The official Pd rule marks a target frame as detected once at least one
    target event crosses the score threshold. A noisy-OR over each
    ``(batch, target ID, time bin)`` group is a smooth counterpart of that
    rule and avoids P4's need to force every positive event to high confidence.
    """
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)
    target_ids = target_ids.reshape(-1).long()
    if locations.ndim != 2 or locations.shape[1] < 4:
        raise ValueError('locations must have shape [N, >=4].')
    if not (
        predictions.numel()
        == labels.numel()
        == target_ids.numel()
        == locations.shape[0]
    ):
        raise ValueError('prediction, label, target-id, and location counts must match.')
    if temporal_bin_size <= 0:
        raise ValueError('temporal_bin_size must be positive.')
    if not 0 < activation_threshold < 1:
        raise ValueError('activation_threshold must be in (0, 1).')
    if activation_temperature <= 0:
        raise ValueError('activation_temperature must be positive.')
    if not 0 < eps < 1:
        raise ValueError('eps must be in (0, 1).')

    event_times = locations[:, 3].long()
    target_mask = (
        (labels > 0.5)
        & (target_ids > 0)
        # Match the official evaluator's open temporal intervals.
        & (torch.remainder(event_times, int(temporal_bin_size)) != 0)
    )
    if not torch.any(target_mask):
        return _zero_loss(predictions), 0, 0

    scores = torch.clamp(predictions[target_mask], min=0, max=1)
    target_ids = target_ids[target_mask]
    batch_ids = locations[target_mask, 0].long()
    time_bins = torch.div(
        event_times[target_mask],
        int(temporal_bin_size),
        rounding_mode='floor',
    )
    target_stride = int(target_ids.max().item()) + 1
    time_stride = int(time_bins.max().item()) + 1
    group_keys = (
        (batch_ids * target_stride + target_ids) * time_stride + time_bins
    )
    _, inverse = torch.unique(group_keys, sorted=True, return_inverse=True)
    group_count = int(inverse.max().item()) + 1
    event_activation = torch.sigmoid(
        (scores - float(activation_threshold)) / float(activation_temperature)
    )
    group_activation = _noisy_or(
        event_activation,
        inverse,
        group_count,
        eps,
    )
    loss = -torch.log(group_activation + float(eps)).mean()
    missed_group_count = int((group_activation < 0.5).sum().item())
    return loss, group_count, missed_group_count


def component_hard_negative_loss(
    predictions,
    labels,
    locations,
    spatial_cell_size,
    temporal_bin_size,
    min_cell_events,
    ratio,
    activation_threshold,
    activation_temperature,
    eps=1e-5,
):
    """Penalize high-confidence background cells likely to become false alarms.

    Challenge 2 counts one false alarm for every spatially connected background
    component in a 50-time-bin frame. This loss groups sampled events into
    small spatial-temporal cells, excludes cells containing labelled target
    events, then mines the most active remaining cells. A differentiable
    noisy-OR makes the loss focus on a cell when one or more of its events
    approach the configured decision threshold.
    """
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)
    if locations.ndim != 2 or locations.shape[1] < 4:
        raise ValueError('locations must have shape [N, >=4].')
    if not predictions.numel() == labels.numel() == locations.shape[0]:
        raise ValueError('prediction, label, and location counts must match.')
    if spatial_cell_size <= 0:
        raise ValueError('spatial_cell_size must be positive.')
    if temporal_bin_size <= 0:
        raise ValueError('temporal_bin_size must be positive.')
    if min_cell_events <= 0:
        raise ValueError('min_cell_events must be positive.')
    if not 0 < ratio <= 1:
        raise ValueError('ratio must be in (0, 1].')
    if not 0 < activation_threshold < 1:
        raise ValueError('activation_threshold must be in (0, 1).')
    if activation_temperature <= 0:
        raise ValueError('activation_temperature must be positive.')
    if not 0 < eps < 1:
        raise ValueError('eps must be in (0, 1).')
    if predictions.numel() == 0:
        return _zero_loss(predictions), 0, 0

    coordinates = locations.long()
    batch_ids = coordinates[:, 0]
    x_coordinates = coordinates[:, 1]
    y_coordinates = coordinates[:, 2]
    time_coordinates = coordinates[:, 3]
    if (
        torch.any(batch_ids < 0)
        or torch.any(x_coordinates < 0)
        or torch.any(y_coordinates < 0)
        or torch.any(time_coordinates < 0)
    ):
        raise ValueError('locations must be non-negative.')

    cell_x = torch.div(
        x_coordinates,
        int(spatial_cell_size),
        rounding_mode='floor',
    )
    cell_y = torch.div(
        y_coordinates,
        int(spatial_cell_size),
        rounding_mode='floor',
    )
    cell_time = torch.div(
        time_coordinates,
        int(temporal_bin_size),
        rounding_mode='floor',
    )
    x_stride = int(cell_x.max().item()) + 1
    y_stride = int(cell_y.max().item()) + 1
    time_stride = int(cell_time.max().item()) + 1
    cell_keys = (
        ((batch_ids * time_stride + cell_time) * y_stride + cell_y) * x_stride
        + cell_x
    )
    _, inverse = torch.unique(cell_keys, sorted=True, return_inverse=True)
    cell_count = int(inverse.max().item()) + 1

    event_counts = predictions.new_zeros(cell_count).scatter_add(
        0,
        inverse,
        torch.ones_like(predictions),
    )
    target_event_counts = predictions.new_zeros(cell_count).scatter_add(
        0,
        inverse,
        (labels > 0.5).to(dtype=predictions.dtype),
    )
    candidate_mask = (
        (target_event_counts == 0)
        & (event_counts >= int(min_cell_events))
    )
    candidate_count = int(candidate_mask.sum().item())
    if candidate_count == 0:
        return _zero_loss(predictions), 0, 0

    scores = torch.clamp(predictions, min=0, max=1)
    event_activation = torch.sigmoid(
        (scores - float(activation_threshold)) / float(activation_temperature)
    )
    # A product of complements represents the chance that at least one event
    # in the cell crosses the soft decision boundary.
    cell_activation = _noisy_or(
        event_activation,
        inverse,
        cell_count,
        eps,
    )
    candidate_activation = cell_activation[candidate_mask]
    hard_count = max(1, int(math.ceil(candidate_count * float(ratio))))
    hard_activation = torch.topk(
        candidate_activation,
        k=hard_count,
        largest=True,
        sorted=False,
    ).values
    loss = -torch.log(1.0 - hard_activation + float(eps)).mean()
    return loss, candidate_count, hard_count
