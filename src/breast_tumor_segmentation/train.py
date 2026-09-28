"""Training and validation utilities for breast tumor segmentation."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from breast_tumor_segmentation.metrics import calculate_dice


@dataclass
class TrainingHistory:
    """Training and validation metrics recorded after each epoch."""

    training_f1: np.ndarray
    training_loss: np.ndarray
    validation_f1: np.ndarray
    validation_loss: np.ndarray


def _prepare_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare a batch for the segmentation model."""
    images = images.unsqueeze(1).to(
        device=device,
        dtype=torch.float32,
    )
    masks = masks.to(
        device=device,
        dtype=torch.int64,
    )

    return images, masks


def train_model(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int = 50,
) -> TrainingHistory:
    """Train a segmentation model and evaluate it after every epoch."""
    training_f1 = np.zeros(epochs, dtype=np.float32)
    training_loss = np.zeros(epochs, dtype=np.float32)
    validation_f1 = np.zeros(epochs, dtype=np.float32)
    validation_loss = np.zeros(epochs, dtype=np.float32)

    for epoch in range(epochs):
        # Training
        model.train()

        training_dice_sum = 0.0
        training_loss_sum = 0.0

        for images, masks in training_loader:
            images, masks = _prepare_batch(
                images,
                masks,
                device,
            )

            optimizer.zero_grad()

            predictions = model(images)
            loss = loss_fn(predictions, masks)

            loss.backward()
            optimizer.step()

            training_loss_sum += loss.item()
            training_dice_sum += calculate_dice(
                predictions.detach(),
                masks,
            ).item()

        training_f1[epoch] = (
            training_dice_sum / len(training_loader)
        )
        training_loss[epoch] = (
            training_loss_sum / len(training_loader)
        )

        # Validation
        model.eval()

        validation_dice_sum = 0.0
        validation_loss_sum = 0.0

        with torch.no_grad():
            for images, masks in validation_loader:
                images, masks = _prepare_batch(
                    images,
                    masks,
                    device,
                )

                predictions = model(images)
                loss = loss_fn(predictions, masks)

                validation_loss_sum += loss.item()
                validation_dice_sum += calculate_dice(
                    predictions,
                    masks,
                ).item()

        validation_f1[epoch] = (
            validation_dice_sum / len(validation_loader)
        )
        validation_loss[epoch] = (
            validation_loss_sum / len(validation_loader)
        )

        print(
            f"Epoch {epoch + 1:03d}/{epochs:03d} | "
            f"train loss: {training_loss[epoch]:.6f} | "
            f"train F1: {training_f1[epoch]:.6f} | "
            f"val loss: {validation_loss[epoch]:.6f} | "
            f"val F1: {validation_f1[epoch]:.6f}"
        )

    return TrainingHistory(
        training_f1=training_f1,
        training_loss=training_loss,
        validation_f1=validation_f1,
        validation_loss=validation_loss,
    )