"""
Confidence calibration head module.
Explicitly predicts detection reliability to reduce false alarms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceHead(nn.Module):
    """Zero-initialized residual logit calibrator.

    Its output is added to the released segmentation logit. The final layer
    starts at exactly zero, so attaching it preserves every prediction before
    the first optimizer step.
    """

    def __init__(self, width, normalization_max_groups=8):
        super().__init__()
        normalization_max_groups = int(normalization_max_groups)
        if normalization_max_groups <= 0:
            raise ValueError('normalization_max_groups must be positive.')
        group_count = next(
            group_count
            for group_count in range(
                min(normalization_max_groups, int(width)),
                0,
                -1,
            )
            if int(width) % group_count == 0
        )
        self.layers = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count, width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, kernel_size=1),
        )

        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, decoder_features):
        """Args:
            decoder_features: Decoder output features [B, C, H, W]

        Returns:
            torch.Tensor: Additive event-logit residual [B, 1, H, W]
        """
        return self.layers(decoder_features)


def confidence_calibration_loss(
    confidence_logits,
    event_logits,
    labels_soft,
    hard_target=True,
):
    """Train a residual confidence calibrator without moving the base model.

    The production path adds this residual to the detached base logit and
    optimizes class-balanced BCE. It can lower background scores without a
    global threshold shift and begins as an exact identity mapping.

    Args:
        confidence_logits: Additive calibration logits [E]
        event_logits: Frozen base event logits [E]
        labels_soft: Ground truth labels [E]
        hard_target: Whether to supervise hard correctness instead of soft
            reliability.

    Returns:
        torch.Tensor: Calibration loss value
    """
    if hard_target:
        calibrated_logits = event_logits.detach() + confidence_logits
        positive = labels_soft > 0.5
        positive_count = positive.sum().clamp(min=1).float()
        negative_count = (~positive).sum().clamp(min=1).float()
        # Class-balanced weights summing to total count, so the loss is
        # scale-invariant in E and stays ~0.69 at init regardless of the
        # positive/negative ratio.
        total = positive_count + negative_count
        weight = torch.where(
            positive,
            0.5 * total / positive_count,
            0.5 * total / negative_count,
        )
        return F.binary_cross_entropy_with_logits(
            calibrated_logits,
            labels_soft,
            weight=weight,
        )
    with torch.no_grad():
        prediction_error = (torch.sigmoid(event_logits) - labels_soft).abs()
        target_confidence = 1.0 - prediction_error

    return F.mse_loss(torch.sigmoid(confidence_logits), target_confidence)
