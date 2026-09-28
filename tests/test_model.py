"""Tests for the U-Net segmentation model."""

import torch

from breast_tumor_segmentation.model import UNet


def test_unet_output_shape() -> None:
    """U-Net should preserve spatial resolution and output two classes."""
    model = UNet(
        in_channels=1,
        out_channels=2,
    )
    model.eval()

    inputs = torch.randn(2, 1, 256, 256)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (2, 2, 256, 256)


def test_unet_output_is_finite() -> None:
    """U-Net output should not contain NaN or infinite values."""
    model = UNet(
        in_channels=1,
        out_channels=2,
    )
    model.eval()

    inputs = torch.randn(2, 1, 256, 256)

    with torch.no_grad():
        outputs = model(inputs)

    assert torch.isfinite(outputs).all()
