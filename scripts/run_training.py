"""Run breast tumor segmentation experiments."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from breast_tumor_segmentation.data import (
    build_dataset,
    create_dataloaders,
    load_metadata,
    split_dataset,
)
from breast_tumor_segmentation.evaluate import evaluate_model
from breast_tumor_segmentation.model import UNet
from breast_tumor_segmentation.preprocessing import build_augmented_dataset
from breast_tumor_segmentation.train import train_model


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 7

EPOCH_OPTIONS = (50, 100)
RUN_SEEDS = tuple(range(20))

LEARNING_RATES = (0.01, 0.0001)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run breast tumor segmentation experiments."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help=(
            "Path to the INbreast Release 1.0 directory containing "
            "AllDICOMs, AllXML, and INbreast.csv."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory in which experiment outputs are stored.",
    )
    return parser.parse_args()


def validate_data_dir(data_dir: Path) -> tuple[Path, Path, Path]:
    """Validate the expected INbreast directory structure."""
    dicom_dir = data_dir / "AllDICOMs"
    xml_dir = data_dir / "AllXML"
    csv_path = data_dir / "INbreast.csv"

    missing_paths = [
        path
        for path in (dicom_dir, xml_dir, csv_path)
        if not path.exists()
    ]

    if missing_paths:
        missing = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "The INbreast dataset directory is incomplete. "
            "Missing required paths:\n"
            f"{missing}"
        )

    return dicom_dir, xml_dir, csv_path


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Run all breast tumor segmentation experiments."""
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()

    dicom_dir, xml_dir, csv_path = validate_data_dir(data_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # -----------------------------------------------------------------
    # Build datasets
    # -----------------------------------------------------------------

    print("Loading INbreast metadata...")
    records = load_metadata(csv_path)

    print(f"Found {len(records)} mammograms.")
    print("Building original dataset...")

    original_dataset = build_dataset(
        records=records,
        dicom_dir=dicom_dir,
        xml_dir=xml_dir,
        image_size=IMAGE_SIZE,
    )

    print(f"Original dataset size: {len(original_dataset)}")

    print("Building augmented dataset...")

    augmented_dataset = build_augmented_dataset(
        original_dataset
    )

    print(f"Augmented dataset size: {len(augmented_dataset)}")

    datasets = {
        "original": original_dataset,
        "augmented": augmented_dataset,
    }

    # -----------------------------------------------------------------
    # Run experiments
    # -----------------------------------------------------------------

    for dataset_name, dataset in datasets.items():
        for learning_rate in LEARNING_RATES:
            for epochs in EPOCH_OPTIONS:
                for seed in RUN_SEEDS:

                    print(
                        "\n"
                        "--------------------------------------------------\n"
                        f"Dataset: {dataset_name} | "
                        f"LR: {learning_rate} | "
                        f"Epochs: {epochs} | "
                        f"Run seed: {seed}\n"
                        "--------------------------------------------------"
                    )

                    # -------------------------------------------------
                    # Reproducible split and DataLoader shuffling
                    # -------------------------------------------------

                    set_seed(seed)

                    training_set, validation_set, testing_set = (
                        split_dataset(
                            dataset,
                            train_fraction=0.7,
                            validation_fraction=0.2,
                            seed=seed,
                        )
                    )

                    print(
                        f"Split | "
                        f"train: {len(training_set)} | "
                        f"validation: {len(validation_set)} | "
                        f"test: {len(testing_set)}"
                    )

                    training_loader, validation_loader, testing_loader = (
                        create_dataloaders(
                            training_set=training_set,
                            validation_set=validation_set,
                            testing_set=testing_set,
                            batch_size=BATCH_SIZE,
                            seed=seed,
                        )
                    )

                    # -------------------------------------------------
                    # Fresh model and optimizer for every run
                    # -------------------------------------------------

                    set_seed(seed)

                    model = UNet(
                        in_channels=1,
                        out_channels=2,
                    ).to(device)

                    loss_fn = nn.CrossEntropyLoss()

                    optimizer = Adam(
                        model.parameters(),
                        lr=learning_rate,
                    )

                    # -------------------------------------------------
                    # Training
                    # -------------------------------------------------

                    history = train_model(
                        model=model,
                        training_loader=training_loader,
                        validation_loader=validation_loader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        device=device,
                        epochs=epochs,
                    )

                    # -------------------------------------------------
                    # Test evaluation
                    # -------------------------------------------------

                    test_loss, test_f1 = evaluate_model(
                        model=model,
                        testing_loader=testing_loader,
                        loss_fn=loss_fn,
                        device=device,
                    )

                    print(f"Test loss: {test_loss:.6f}")
                    print(f"Test F1/Dice: {test_f1:.6f}")

                    # -------------------------------------------------
                    # Save results
                    # -------------------------------------------------

                    run_results_dir = (
                        results_dir
                        / dataset_name
                        / f"lr_{learning_rate}"
                        / f"epochs_{epochs}"
                        / f"run_{seed}"
                    )

                    run_results_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    np.save(
                        run_results_dir / "training_f1.npy",
                        history.training_f1,
                    )

                    np.save(
                        run_results_dir / "training_loss.npy",
                        history.training_loss,
                    )

                    np.save(
                        run_results_dir / "validation_f1.npy",
                        history.validation_f1,
                    )

                    np.save(
                        run_results_dir / "validation_loss.npy",
                        history.validation_loss,
                    )

                    np.save(
                        run_results_dir / "test_metrics.npy",
                        np.array(
                            [test_loss, test_f1],
                            dtype=np.float32,
                        ),
                    )

    print(
        "\nAll experiments completed. "
        f"Results saved to: {results_dir}"
    )


if __name__ == "__main__":
    main()