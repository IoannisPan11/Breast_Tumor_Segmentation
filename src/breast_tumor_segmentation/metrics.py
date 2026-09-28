"""Evaluation metrics for breast tumor segmentation."""

import torch


def calculate_dice(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 2,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute macro Dice across background and tumor classes for one batch."""
    predicted_classes = torch.argmax(predictions, dim=1)

    dice_scores = []

    for class_index in range(num_classes):
        predicted_mask = predicted_classes == class_index
        target_mask = targets == class_index

        intersection = (predicted_mask & target_mask).sum().float()
        denominator = predicted_mask.sum().float() + target_mask.sum().float()

        dice = (2.0 * intersection + eps) / (denominator + eps)
        dice_scores.append(dice)

    return torch.stack(dice_scores).mean()