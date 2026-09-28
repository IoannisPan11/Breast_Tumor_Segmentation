"""Evaluation utilities for breast tumor segmentation."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from breast_tumor_segmentation.metrics import calculate_dice


def evaluate_model(
    model: nn.Module,
    testing_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate a trained model on the test set.

    Returns:
        Mean test loss and mean Dice/F1 score across batches.
    """
    model.eval()

    test_loss_sum = 0.0
    test_dice_sum = 0.0

    with torch.no_grad():
        for images, masks in testing_loader:
            images = images.unsqueeze(1).to(
                device=device,
                dtype=torch.float32,
            )
            masks = masks.to(
                device=device,
                dtype=torch.int64,
            )

            predictions = model(images)
            loss = loss_fn(predictions, masks)

            test_loss_sum += loss.item()
            test_dice_sum += calculate_dice(
                predictions,
                masks,
            ).item()

    mean_test_loss = test_loss_sum / len(testing_loader)
    mean_test_f1 = test_dice_sum / len(testing_loader)

    return mean_test_loss, mean_test_f1