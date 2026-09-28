"""Data loading and dataset utilities for the INbreast dataset."""

import csv
from pathlib import Path

import numpy as np
import pydicom
import torch
from inbreastxmlparser.annotation import INbreastAnnotation
from skimage.draw import polygon
from torch.utils.data import DataLoader, Dataset

from breast_tumor_segmentation.preprocessing import (
    crop_background,
    normalize_image,
    resize_image_and_mask,
)


class INBreastDataset(Dataset):
    """PyTorch dataset containing preprocessed mammograms and masks."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.data[index, :, :, 0]).float()
        mask = torch.from_numpy(self.data[index, :, :, 1]).long()

        return image, mask


def load_metadata(csv_path: Path) -> list[dict[str, str]]:
    """Load mammogram identifiers and metadata from the INbreast CSV."""
    records = []

    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file, delimiter=";")

        for row in reader:
            records.append(
                {
                    "filename": row["File Name"].strip(),
                    "laterality": row["Laterality"].strip(),
                    "view": row["View"].strip(),
                    "acr": row["ACR"].strip(),
                    "birads": row["Bi-Rads"].strip(),
                }
            )

    return records


def build_mass_mask(
    annotation: INbreastAnnotation,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Build a mass mask while clipping rasterized ROIs to image bounds."""
    mask = np.zeros(image_shape, dtype=np.uint8)

    if not Path(annotation.xml_path).exists():
        return mask

    rois, _ = annotation.parse_XML(annotation.xml_path)
    image_height, image_width = image_shape

    for roi in rois:
        roi_info = annotation.get_roi_info(roi)

        if roi_info["roi_type"] != "Mass":
            continue

        r_poly, c_poly = annotation.create_polygon_lists(
            annotation.mask,
            roi_info["points"],
        )
        rr, cc = polygon(r_poly, c_poly)

        valid = (
            (rr >= 0)
            & (rr < image_height)
            & (cc >= 0)
            & (cc < image_width)
        )

        mask[rr[valid], cc[valid]] = 1

    return mask


def load_image_and_mask(
    image_name: str,
    dicom_dir: Path,
    xml_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a DICOM mammogram and its corresponding segmentation mask."""
    dicom_matches = list(dicom_dir.glob(f"{image_name}_*.dcm"))

    if len(dicom_matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one DICOM for image ID {image_name}, "
            f"found {len(dicom_matches)}."
        )

    image = pydicom.dcmread(dicom_matches[0]).pixel_array

    annotation = INbreastAnnotation(
        str(xml_dir),
        image_name,
        image.shape,
    )
    mask = build_mass_mask(
        annotation=annotation,
        image_shape=image.shape,
    )

    return image, mask


def build_dataset(
    records: list[dict[str, str]],
    dicom_dir: Path,
    xml_dir: Path,
    image_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Load and preprocess the original INbreast mammograms and masks."""
    dataset = np.zeros(
        (
            len(records),
            image_size[1],
            image_size[0],
            2,
        ),
        dtype=np.float32,
    )

    for index, record in enumerate(records):
        image, mask = load_image_and_mask(
            record["filename"],
            dicom_dir,
            xml_dir,
        )

        # Per-image normalization to [0, 255], as in the thesis pipeline.
        image = normalize_image(image)
        image = (image * 255).astype(np.uint8)

        # Remove zero-valued borders and apply the same crop to the mask.
        cropped_image, left, right, top, bottom = crop_background(image)
        cropped_mask = mask[top:bottom, left:right]

        # Resize both channels to the U-Net input resolution.
        resized_image, resized_mask = resize_image_and_mask(
            cropped_image,
            cropped_mask,
            image_size,
        )

        dataset[index, :, :, 0] = resized_image
        dataset[index, :, :, 1] = resized_mask

    # Global normalization of the 410-image original dataset to [0, 1].
    # In the thesis pipeline this happens before any augmentation is created.
    image_channel = dataset[:, :, :, 0]
    global_min = image_channel.min()
    global_max = image_channel.max()

    if global_max == global_min:
        raise ValueError(
            "Cannot normalize dataset because all image values are identical."
        )

    dataset[:, :, :, 0] = (
        image_channel - global_min
    ) / (
        global_max - global_min
    )

    return dataset


def split_dataset(
    dataset: np.ndarray,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split data into training, validation, and test sets."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError(
            "Training and validation fractions must be greater than zero."
        )

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "Training and validation fractions must sum to less than 1."
        )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(dataset))
    shuffled_dataset = dataset[permutation]

    train_end = int(train_fraction * len(shuffled_dataset))
    validation_end = train_end + int(
        validation_fraction * len(shuffled_dataset)
    )

    training_set = shuffled_dataset[:train_end]
    validation_set = shuffled_dataset[train_end:validation_end]
    testing_set = shuffled_dataset[validation_end:]

    return training_set, validation_set, testing_set


def create_dataloaders(
    training_set: np.ndarray,
    validation_set: np.ndarray,
    testing_set: np.ndarray,
    batch_size: int = 7,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create training, validation, and test DataLoaders for one run."""
    training_generator = None

    if seed is not None:
        training_generator = torch.Generator()
        training_generator.manual_seed(seed)

    training_loader = DataLoader(
        INBreastDataset(training_set),
        batch_size=batch_size,
        shuffle=True,
        generator=training_generator,
    )

    validation_loader = DataLoader(
        INBreastDataset(validation_set),
        batch_size=batch_size,
        shuffle=False,
    )

    testing_loader = DataLoader(
        INBreastDataset(testing_set),
        batch_size=batch_size,
        shuffle=False,
    )

    return training_loader, validation_loader, testing_loader