"""Analyze repeated breast tumor segmentation experiments."""

import argparse
from pathlib import Path

import numpy as np
from emv import utils


DATASET_OPTIONS = ("original", "augmented")
LEARNING_RATES = (0.01, 0.0001)
EPOCH_OPTIONS = (50, 100)
RUN_SEEDS = tuple(range(20))

METRIC_FILES = (
    "training_f1",
    "training_loss",
    "validation_f1",
    "validation_loss",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze repeated breast tumor segmentation experiments."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing the saved experiment outputs.",
    )
    return parser.parse_args()


def configuration_dir(
    results_dir: Path,
    dataset_name: str,
    learning_rate: float,
    epochs: int,
) -> Path:
    """Return the directory containing the 20 runs of one configuration."""
    return (
        results_dir
        / dataset_name
        / f"lr_{learning_rate}"
        / f"epochs_{epochs}"
    )


def load_epoch_metric(
    config_dir: Path,
    metric_name: str,
    epochs: int,
) -> np.ndarray:
    """Load one epoch-wise metric from all repeated runs."""
    values = []

    for seed in RUN_SEEDS:
        metric_path = config_dir / f"run_{seed}" / f"{metric_name}.npy"
        metric = np.load(metric_path)

        if metric.shape != (epochs,):
            raise ValueError(
                f"Unexpected shape for {metric_path}: "
                f"expected {(epochs,)}, got {metric.shape}."
            )

        values.append(metric)

    return np.stack(values, axis=0)


def load_test_metrics(config_dir: Path) -> np.ndarray:
    """Load test loss and test F1/Dice from all repeated runs."""
    values = []

    for seed in RUN_SEEDS:
        metric_path = config_dir / f"run_{seed}" / "test_metrics.npy"
        metric = np.load(metric_path)

        if metric.shape != (2,):
            raise ValueError(
                f"Unexpected shape for {metric_path}: "
                f"expected {(2,)}, got {metric.shape}."
            )

        values.append(metric)

    return np.stack(values, axis=0)


def plot_metric_with_emv(
    values: np.ndarray,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot repeated-run metrics using the original thesis EMV analysis."""
    iterations, epochs = values.shape

    # Original thesis shape: (epochs, iterations, metrics).
    emv_values = values.T[:, :, np.newaxis]

    gen_args = {
        "num_of_timepoints": epochs,
        "num_of_iterations": iterations,
        "num_of_metrics": 1,
        "metric_id": 0,
        "window": 0.8,
    }

    fig_args = {
        "fig_path": str(output_path),
        "xlabel": "Epochs",
        "ylabel": ylabel,
        "font_size": 14,
        "y_min": 0,
        "y_max": 1,
    }

    ci_elems = utils.compute_ci(
        emv_values,
        gen_args,
    )

    utils.plot_metrics_ci(
        ci_elems,
        gen_args["num_of_timepoints"],
        fig_args,
    )


def analyze_configuration(
    results_dir: Path,
    figures_dir: Path,
    dataset_name: str,
    learning_rate: float,
    epochs: int,
) -> None:
    """Analyze the 20 repeated runs of one experimental configuration."""
    config_dir = configuration_dir(
        results_dir=results_dir,
        dataset_name=dataset_name,
        learning_rate=learning_rate,
        epochs=epochs,
    )

    metrics = {
        metric_name: load_epoch_metric(
            config_dir=config_dir,
            metric_name=metric_name,
            epochs=epochs,
        )
        for metric_name in METRIC_FILES
    }

    test_metrics = load_test_metrics(config_dir)

    test_loss = test_metrics[:, 0]
    test_f1 = test_metrics[:, 1]

    test_loss_mean = float(test_loss.mean())
    test_loss_std = float(test_loss.std(ddof=1))
    test_f1_mean = float(test_f1.mean())
    test_f1_std = float(test_f1.std(ddof=1))

    figure_dir = (
        figures_dir
        / dataset_name
        / f"lr_{learning_rate}"
        / f"epochs_{epochs}"
    )
    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_metric_with_emv(
        metrics["training_f1"],
        ylabel="Training F1",
        output_path=figure_dir / "training_f1.png",
    )

    plot_metric_with_emv(
        metrics["training_loss"],
        ylabel="Training Loss",
        output_path=figure_dir / "training_loss.png",
    )

    plot_metric_with_emv(
        metrics["validation_f1"],
        ylabel="Validation F1",
        output_path=figure_dir / "validation_f1.png",
    )

    plot_metric_with_emv(
        metrics["validation_loss"],
        ylabel="Validation Loss",
        output_path=figure_dir / "validation_loss.png",
    )

    print(
        f"{dataset_name:9s} | "
        f"lr={learning_rate:<6g} | "
        f"epochs={epochs:3d} | "
        f"test F1/Dice={test_f1_mean:.6f} ± {test_f1_std:.6f} | "
        f"test loss={test_loss_mean:.6f} ± {test_loss_std:.6f}"
    )


def main() -> None:
    """Analyze all eight experimental configurations."""
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    figures_dir = results_dir / "figures"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Summary across 20 runs per configuration:\n")

    for dataset_name in DATASET_OPTIONS:
        for learning_rate in LEARNING_RATES:
            for epochs in EPOCH_OPTIONS:
                analyze_configuration(
                    results_dir=results_dir,
                    figures_dir=figures_dir,
                    dataset_name=dataset_name,
                    learning_rate=learning_rate,
                    epochs=epochs,
                )

    print(
        f"\nFigures saved to: {figures_dir.resolve()}"
    )


if __name__ == "__main__":
    main()