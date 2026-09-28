"""Tests for mammogram preprocessing and augmentation utilities."""

import numpy as np

from breast_tumor_segmentation.preprocessing import (
    build_augmented_dataset,
    normalize_image,
    resize_image_and_mask,
    rotate_180,
)


def test_normalize_image_range() -> None:
    """Min-max normalization should map image values to [0, 1]."""
    image = np.array(
        [
            [10, 20],
            [30, 40],
        ],
        dtype=np.float32,
    )

    normalized = normalize_image(image)

    assert normalized.dtype == np.float32
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(normalized.max(), 1.0)


def test_normalize_constant_image_returns_zeros() -> None:
    """A constant image should normalize safely to zeros."""
    image = np.full(
        (4, 4),
        5,
        dtype=np.float32,
    )

    normalized = normalize_image(image)

    assert np.all(normalized == 0.0)


def test_resize_image_and_mask_shape() -> None:
    """Image and mask should be resized to the requested resolution."""
    image = np.ones(
        (100, 80),
        dtype=np.uint8,
    )
    mask = np.zeros(
        (100, 80),
        dtype=np.uint8,
    )
    mask[20:40, 30:50] = 1

    resized_image, resized_mask = resize_image_and_mask(
        image,
        mask,
        size=(256, 256),
    )

    assert resized_image.shape == (256, 256)
    assert resized_mask.shape == (256, 256)


def test_rotate_180_preserves_shape_and_binary_mask() -> None:
    """A 180-degree rotation should preserve square shape and mask classes."""
    image = np.zeros(
        (256, 256),
        dtype=np.float32,
    )
    mask = np.zeros(
        (256, 256),
        dtype=np.float32,
    )

    image[20:50, 30:60] = 1.0
    mask[20:50, 30:60] = 1.0

    rotated_image, rotated_mask = rotate_180(
        image,
        mask,
    )

    assert rotated_image.shape == (256, 256)
    assert rotated_mask.shape == (256, 256)
    assert set(np.unique(rotated_mask)).issubset({0.0, 1.0})


def test_augmented_dataset_has_six_times_original_size() -> None:
    """Thesis augmentation pipeline should expand N samples to 6N."""
    dataset = np.zeros(
        (2, 256, 256, 2),
        dtype=np.float32,
    )

    dataset[0, 50:150, 50:150, 0] = 0.5
    dataset[0, 80:120, 80:120, 1] = 1.0

    dataset[1, 30:180, 40:160, 0] = 0.8
    dataset[1, 90:130, 90:130, 1] = 1.0

    augmented = build_augmented_dataset(dataset)

    assert augmented.shape == (12, 256, 256, 2)
    assert augmented.dtype == np.float32
    assert np.isfinite(augmented).all()
    assert set(np.unique(augmented[:, :, :, 1])).issubset({0.0, 1.0})