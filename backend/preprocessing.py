"""
preprocessing.py
-----------------
Image preprocessing pipeline for the oil spill detection system.

This module reproduces - exactly, numerically - the validated HSV /
GaussianBlur / morphology pipeline from the original project, but breaks
it into small, testable, reusable functions so it can be composed with
the new stitching and MOSDAC-ingestion stages.

Pipeline stages implemented:
    * image validation
    * resize
    * denoising
    * Gaussian blur
    * contrast enhancement (CLAHE) + histogram equalization
    * normalization
    * HSV conversion
    * morphological open/close
    * ROI extraction (vignette-edge cropping)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from backend import config
from backend.utils import get_logger

logger = get_logger(__name__)


class ImageValidationError(Exception):
    """Raised when an input image fails basic sanity checks."""


@dataclass
class PreprocessedResult:
    """Container for every intermediate artifact produced by the pipeline."""

    original: np.ndarray
    resized: np.ndarray
    denoised: np.ndarray
    blurred: np.ndarray
    contrast_enhanced: np.ndarray
    hsv: np.ndarray
    oil_mask_raw: np.ndarray
    oil_mask_clean: np.ndarray
    roi_mask: np.ndarray
    display_preprocessed: np.ndarray  # BGR image suitable for saving/showing


def validate_image(image: np.ndarray, source_name: str = "image") -> None:
    """
    Run basic sanity checks on a loaded image array.

    Raises
    ------
    ImageValidationError
        If the image is None, empty, or has an unexpected number of channels.
    """
    if image is None:
        raise ImageValidationError(f"'{source_name}' could not be read (None).")
    if image.size == 0:
        raise ImageValidationError(f"'{source_name}' is empty.")
    if image.ndim not in (2, 3):
        raise ImageValidationError(
            f"'{source_name}' has unexpected dimensionality: {image.ndim}."
        )
    height, width = image.shape[:2]
    if height < 16 or width < 16:
        raise ImageValidationError(
            f"'{source_name}' is too small to analyze ({width}x{height})."
        )


def load_image(filepath: str) -> np.ndarray:
    """Load an image from disk (BGR, as OpenCV expects) with validation."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No such file: {filepath}")

    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    validate_image(image, source_name=os.path.basename(filepath))
    return image


def resize_image(
    image: np.ndarray, target_size: Tuple[int, int] = config.MODEL_INPUT_SIZE
) -> np.ndarray:
    """Resize an image to ``target_size`` using area interpolation (best for shrinking)."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Remove sensor / compression noise while preserving edges."""
    return cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)


def apply_gaussian_blur(image: np.ndarray) -> np.ndarray:
    """Smooth out water-surface texture so wave ripples don't look like oil edges."""
    cfg = config.PREPROCESSING
    return cv2.GaussianBlur(image, cfg.gaussian_kernel, cfg.gaussian_sigma)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE (adaptive histogram equalization) on the luminance channel."""
    cfg = config.PREPROCESSING
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_tile_grid_size
    )
    l_equalized = clahe.apply(l_channel)

    merged = cv2.merge((l_equalized, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def histogram_equalize_grayscale(image: np.ndarray) -> np.ndarray:
    """Standard global histogram equalization on a grayscale copy (diagnostic aid)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Scale pixel values to the [0, 1] float range (used before feeding the CNN)."""
    return image.astype(np.float32) / 255.0


def convert_to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to HSV color space."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def threshold_oil_regions(hsv_image: np.ndarray) -> np.ndarray:
    """
    Threshold the HSV image for dark, low-saturation regions characteristic
    of oil slicks on open water. Ranges match the values validated during
    model development and must stay in sync with config.PREPROCESSING.
    """
    cfg = config.PREPROCESSING
    lower_oil = np.array(cfg.hsv_lower_oil)
    upper_oil = np.array(cfg.hsv_upper_oil)
    return cv2.inRange(hsv_image, lower_oil, upper_oil)


def clean_mask_morphology(mask: np.ndarray) -> np.ndarray:
    """Remove speckle noise (OPEN) then fill small holes (CLOSE) in a binary mask."""
    cfg = config.PREPROCESSING
    kernel = np.ones(cfg.morph_kernel_size, np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def build_roi_mask(height: int, width: int) -> np.ndarray:
    """
    Build a binary ROI mask that excludes the top/bottom vignette-prone
    edges of the frame (a common source of false positives in raw
    satellite/aerial captures).
    """
    cfg = config.PREPROCESSING
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    margin = cfg.roi_vertical_margin
    roi_mask[int(height * margin) : int(height * (1 - margin)), :] = 255
    return roi_mask


def run_preprocessing_pipeline(image: np.ndarray) -> PreprocessedResult:
    """
    Execute the full preprocessing pipeline on a single BGR image and
    return every intermediate artifact (used both for prediction and for
    dashboard visualization).
    """
    validate_image(image, source_name="input")
    height, width = image.shape[:2]

    denoised = denoise_image(image)
    blurred = apply_gaussian_blur(denoised)
    contrast_enhanced = enhance_contrast(blurred)

    hsv = convert_to_hsv(blurred)
    oil_mask_raw = threshold_oil_regions(hsv)
    oil_mask_clean = clean_mask_morphology(oil_mask_raw)

    roi_mask = build_roi_mask(height, width)
    oil_mask_clean = cv2.bitwise_and(oil_mask_clean, roi_mask)

    # A human-viewable "preprocessed" image for the dashboard: contrast
    # enhanced + blurred, representing what the model effectively "sees".
    display_preprocessed = contrast_enhanced

    logger.debug("Preprocessing pipeline completed for %dx%d image.", width, height)

    return PreprocessedResult(
        original=image,
        resized=resize_image(image),
        denoised=denoised,
        blurred=blurred,
        contrast_enhanced=contrast_enhanced,
        hsv=hsv,
        oil_mask_raw=oil_mask_raw,
        oil_mask_clean=oil_mask_clean,
        roi_mask=roi_mask,
        display_preprocessed=display_preprocessed,
    )
