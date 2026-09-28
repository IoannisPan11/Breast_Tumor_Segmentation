"""Tests for breast tumor segmentation metrics."""

import torch

from breast_tumor_segmentation.metrics import calculate_dice


def test_dice_is_one_for_perfect_prediction() -> None:
    """Macro Dice should equal one for a perfect two-class prediction."""
    targets = torch.tensor(
        [
            [
                [0, 0],
                [1, 1],
            ]
        ],
        dtype=torch.int64,
    )

    predictions = torch.tensor(
        [
            [
                [
                    [10.0, 10.0],
                    [0.0, 0.0],
                ],
                [
                    [0.0, 0.0],
                    [10.0, 10.0],
                ],
            ]
        ]
    )

    dice = calculate_dice(predictions, targets)

    assert torch.isclose(dice, torch.tensor(1.0))


def test_dice_matches_expected_macro_value() -> None:
    """Macro Dice should average the Dice scores of both classes."""
    targets = torch.tensor(
        [
            [
                [0, 0],
                [1, 1],
            ]
        ],
        dtype=torch.int64,
    )

    predicted_classes = torch.tensor(
        [
            [
                [0, 1],
                [1, 1],
            ]
        ],
        dtype=torch.int64,
    )

    predictions = torch.nn.functional.one_hot(
        predicted_classes,
        num_classes=2,
    ).permute(0, 3, 1, 2).float()

    dice = calculate_dice(predictions, targets)

    # Background Dice = 2 / 3
    # Tumor Dice = 4 / 5
    # Macro Dice = ((2 / 3) + (4 / 5)) / 2 = 11 / 15
    expected = torch.tensor(11.0 / 15.0)

    assert torch.isclose(dice, expected)


def test_dice_is_finite_when_class_is_absent() -> None:
    """Dice should remain finite when a class is absent from both masks."""
    targets = torch.zeros(
        (1, 2, 2),
        dtype=torch.int64,
    )

    predictions = torch.zeros(
        (1, 2, 2, 2),
        dtype=torch.float32,
    )
    predictions[:, 0] = 1.0

    dice = calculate_dice(predictions, targets)

    assert torch.isfinite(dice)
    assert torch.isclose(dice, torch.tensor(1.0))