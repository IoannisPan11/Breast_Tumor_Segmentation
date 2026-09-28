"""Preprocessing and augmentation utilities for mammogram images."""

import cv2
import numpy as np
from skimage.exposure import adjust_gamma, equalize_hist
from skimage.transform import rotate


def crop_background(image: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    """Remove zero-valued borders from a mammogram."""
    left = 0
    right = image.shape[1] - 1
    top = 0
    bottom = image.shape[0] - 1

    while np.sum(image[:, left]) == 0:
        left += 1
    while np.sum(image[:, right]) == 0:
        right -= 1
    while np.sum(image[top, :]) == 0:
        top += 1
    while np.sum(image[bottom, :]) == 0:
        bottom -= 1

    cropped = image[top:bottom, left:right]

    return cropped, left, right, top, bottom


def resize_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    size: tuple[int, int] = (256, 256),
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a mammogram and its segmentation mask."""
    resized_image = cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA,
    )
    resized_mask = cv2.resize(
        mask,
        size,
        interpolation=cv2.INTER_AREA,
    )

    return resized_image, resized_mask


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Min-max normalize an image to the [0, 1] range."""
    image_min = image.min()
    image_max = image.max()

    if image_max == image_min:
        return np.zeros_like(image, dtype=np.float32)

    return (
        (image - image_min) / (image_max - image_min)
    ).astype(np.float32)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization."""
    return equalize_hist(image)


def gamma_correction(
    image: np.ndarray,
    gamma: float = 0.5,
    gain: float = 1.0,
) -> np.ndarray:
    """Apply gamma correction."""
    return adjust_gamma(
        image,
        gamma=gamma,
        gain=gain,
    )


def rotate_180(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate an image and its segmentation mask by 180 degrees."""
    rotated_image = rotate(
        image,
        angle=180,
        resize=True,
    )
    rotated_mask = rotate(
        mask,
        angle=180,
        resize=True,
    )

    return rotated_image, rotated_mask


def build_augmented_dataset(dataset: np.ndarray) -> np.ndarray:
    """Create the 2,460-sample augmented dataset used in the thesis."""
    histogram_dataset = dataset.copy()
    gamma_dataset = dataset.copy()

    for index in range(len(dataset)):
        histogram_dataset[index, :, :, 0] = histogram_equalization(
            dataset[index, :, :, 0]
        )
        gamma_dataset[index, :, :, 0] = gamma_correction(
            dataset[index, :, :, 0]
        )

    original_histogram_gamma = np.concatenate(
        [dataset, histogram_dataset, gamma_dataset],
        axis=0,
    )

    rotated_dataset = np.empty_like(original_histogram_gamma)

    for index in range(len(original_histogram_gamma)):
        rotated_image, rotated_mask = rotate_180(
            original_histogram_gamma[index, :, :, 0],
            original_histogram_gamma[index, :, :, 1],
        )

        rotated_dataset[index, :, :, 0] = rotated_image
        rotated_dataset[index, :, :, 1] = rotated_mask

    final_dataset = np.concatenate(
        [original_histogram_gamma, rotated_dataset],
        axis=0,
    )

    return final_dataset