"""Tests for dataset splitting and PyTorch data loading."""

import numpy as np
import torch

from breast_tumor_segmentation.data import (
    INBreastDataset,
    create_dataloaders,
    split_dataset,
)


def create_test_dataset(num_samples: int = 100) -> np.ndarray:
    """Create a small deterministic array for data-pipeline unit tests."""
    dataset = np.zeros(
        (num_samples, 16, 16, 2),
        dtype=np.float32,
    )

    for index in range(num_samples):
        dataset[index, :, :, 0] = index / num_samples
        dataset[index, :, :, 1] = index % 2

    return dataset


def test_inbreast_dataset_returns_expected_types() -> None:
    """Dataset should return float images and integer segmentation masks."""
    dataset = create_test_dataset(num_samples=4)
    torch_dataset = INBreastDataset(dataset)

    image, mask = torch_dataset[0]

    assert image.shape == (16, 16)
    assert mask.shape == (16, 16)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.int64


def test_split_dataset_uses_expected_proportions() -> None:
    """A 70/20/10 split should produce the expected sample counts."""
    dataset = create_test_dataset(num_samples=100)

    training_set, validation_set, testing_set = split_dataset(
        dataset,
        train_fraction=0.7,
        validation_fraction=0.2,
        seed=0,
    )

    assert len(training_set) == 70
    assert len(validation_set) == 20
    assert len(testing_set) == 10


def test_split_dataset_is_reproducible() -> None:
    """Using the same seed should reproduce the same dataset split."""
    dataset = create_test_dataset(num_samples=100)

    first_split = split_dataset(
        dataset,
        train_fraction=0.7,
        validation_fraction=0.2,
        seed=7,
    )

    second_split = split_dataset(
        dataset,
        train_fraction=0.7,
        validation_fraction=0.2,
        seed=7,
    )

    for first, second in zip(first_split, second_split):
        assert np.array_equal(first, second)


def test_different_seeds_produce_different_splits() -> None:
    """Different seeds should produce different shuffled training sets."""
    dataset = create_test_dataset(num_samples=100)

    first_training_set, _, _ = split_dataset(
        dataset,
        seed=1,
    )

    second_training_set, _, _ = split_dataset(
        dataset,
        seed=2,
    )

    assert not np.array_equal(
        first_training_set,
        second_training_set,
    )


def test_create_dataloaders_returns_expected_batch_shapes() -> None:
    """DataLoaders should return batches with thesis-compatible shapes."""
    dataset = create_test_dataset(num_samples=100)

    training_set, validation_set, testing_set = split_dataset(
        dataset,
        seed=0,
    )

    training_loader, validation_loader, testing_loader = create_dataloaders(
        training_set,
        validation_set,
        testing_set,
        batch_size=7,
        seed=0,
    )

    images, masks = next(iter(training_loader))

    assert images.shape == (7, 16, 16)
    assert masks.shape == (7, 16, 16)
    assert images.dtype == torch.float32
    assert masks.dtype == torch.int64

    assert len(validation_loader.dataset) == 20
    assert len(testing_loader.dataset) == 10


def test_training_dataloader_is_reproducible() -> None:
    """The same seed should reproduce the training shuffle sequence."""
    dataset = create_test_dataset(num_samples=100)

    training_set, validation_set, testing_set = split_dataset(
        dataset,
        seed=0,
    )

    first_loader, _, _ = create_dataloaders(
        training_set,
        validation_set,
        testing_set,
        batch_size=7,
        seed=42,
    )

    second_loader, _, _ = create_dataloaders(
        training_set,
        validation_set,
        testing_set,
        batch_size=7,
        seed=42,
    )

    first_images, first_masks = next(iter(first_loader))
    second_images, second_masks = next(iter(second_loader))

    assert torch.equal(first_images, second_images)
    assert torch.equal(first_masks, second_masks)