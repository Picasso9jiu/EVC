"""Training loss that balances positive supervision across target frames."""

import torch
import torch.nn.functional as functional


def target_frame_balanced_positive_loss(
    predictions,
    labels,
    target_ids,
    locations,
    temporal_bin_size,
    eps=1e-5,
    from_logits=False,
):
    """Average positive BCE equally over official target-time groups.

    Semantic labels are event-level, while Challenge 2 Pd gives the same
    value to every ``(target ID, temporal bin)`` group. This helper keeps
    gradients event-level but first averages events within each target frame,
    then averages the target-frame losses. Events on bin boundaries are
    excluded because the official evaluator excludes them from Pd as well.

    ``predictions`` are probabilities by default for compatibility with the
    original STC loss path. Set ``from_logits`` for the temporal-memory
    trainer so its auxiliary positive BCE remains numerically stable for
    low-confidence target windows.
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
        raise ValueError(
            'Prediction, label, target-id, and location counts must match.'
        )
    if int(temporal_bin_size) <= 0:
        raise ValueError('temporal_bin_size must be positive.')
    if float(eps) <= 0:
        raise ValueError('eps must be positive.')

    event_times = locations[:, 3].long()
    target_mask = (
        (labels > 0.5)
        & (target_ids > 0)
        & (torch.remainder(event_times, int(temporal_bin_size)) != 0)
    )
    if not torch.any(target_mask):
        return predictions.sum() * 0, 0

    if from_logits:
        event_losses = functional.softplus(-predictions[target_mask])
    else:
        selected_predictions = torch.clamp(
            predictions[target_mask], min=0, max=1
        )
        event_losses = -torch.log(selected_predictions + float(eps))
    selected_target_ids = target_ids[target_mask]
    batch_ids = locations[target_mask, 0].long()
    time_bins = torch.div(
        event_times[target_mask],
        int(temporal_bin_size),
        rounding_mode='floor',
    )

    target_stride = int(selected_target_ids.max().item()) + 1
    time_stride = int(time_bins.max().item()) + 1
    group_keys = (
        (batch_ids * target_stride + selected_target_ids) * time_stride
        + time_bins
    )
    group_keys, order = torch.sort(group_keys)
    event_losses = event_losses[order]
    _, counts = torch.unique_consecutive(group_keys, return_counts=True)

    group_count = int(counts.numel())
    # ``scatter_add`` has no deterministic CUDA implementation in the
    # project's PyTorch version. Consecutive groups let prefix sums produce
    # the same group totals without atomic accumulation.
    prefix_sums = torch.cat((
        event_losses.new_zeros(1),
        torch.cumsum(event_losses, dim=0),
    ))
    group_ends = torch.cumsum(counts, dim=0)
    group_starts = group_ends - counts
    group_sums = (
        prefix_sums.index_select(0, group_ends)
        - prefix_sums.index_select(0, group_starts)
    )
    group_means = group_sums / counts.to(dtype=event_losses.dtype)
    return group_means.mean(), group_count
