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
from typing import Optional, Tuple

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
    land_mask: np.ndarray | None = None


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


def threshold_oil_regions(
    hsv_image: np.ndarray, bgr_image: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Threshold the HSV image for dark, low-saturation regions characteristic
    of oil slicks on open water.
    
    Uses adaptive ambient water contrast gating:
    - In deep oceanic waters (e.g. test.jpg, where ambient water peak is <= 65),
      the upper V threshold is strictly clamped to <= 50 to prevent dark clean ocean
      from falsely inflating the segmentation mask.
    - In moderate or bright marine waters (e.g. oil.jpg, a.jpeg, where ambient water
      peak is >= 80 and clean water V >= 71), the upper V threshold is safely set to 65
      to capture all dark slick bands (including edge-diluted and patchy slicks) without
      touching clean seawater.
    - In SAR imagery (clear_ocean_baseline.png), default V <= 50 is used.
    """
    cfg = config.PREPROCESSING
    lower_oil = np.array(cfg.hsv_lower_oil)
    upper_v = cfg.hsv_upper_oil[2]

    if bgr_image is not None and is_optical_marine_image(bgr_image):
        b, g, r = bgr_image[:, :, 0], bgr_image[:, :, 1], bgr_image[:, :, 2]
        water_mask = (b > r + 3) & (g > r)
        v = hsv_image[:, :, 2]
        water_v = v[water_mask]
        if len(water_v) > 200:
            hist, bin_edges = np.histogram(water_v, bins=51, range=(0, 255))
            peak_idx = int(np.argmax(hist))
            water_peak = 0.5 * (bin_edges[peak_idx] + bin_edges[peak_idx + 1])
            if water_peak <= 65:
                # Deep dark ocean: clean water begins around 54; clamp strictly <= 50
                upper_v = min(50, max(42, int(water_peak - 6)))
            elif water_peak < 80:
                upper_v = min(58, max(48, int(water_peak - 12)))
            else:
                # Bright / moderate water: clean water is >= 71; upper V = 65 safely captures all slicks
                upper_v = 65

    upper_oil = np.array([cfg.hsv_upper_oil[0], cfg.hsv_upper_oil[1], upper_v])
    return cv2.inRange(hsv_image, lower_oil, upper_oil)


def clean_mask_morphology(mask: np.ndarray) -> np.ndarray:
    """Remove speckle noise (OPEN) then fill small holes (CLOSE) in a binary mask."""
    h, w = mask.shape[:2]
    ksize = (3, 3) if min(h, w) < 400 else config.PREPROCESSING.morph_kernel_size
    kernel = np.ones(ksize, np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def detect_nodata_collar(image: np.ndarray) -> np.ndarray:
    """
    Identify synthetic zero-padding / NoData collar borders common in
    reprojected satellite SAR scenes. Flat, uniform boundary padding is
    masked out so it is not mistaken for a low-backscatter oil slick.
    """
    height, width = image.shape[:2]
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    is_padding = (
        (b <= 15)
        & (g <= 15)
        & (r <= 50)
        & (np.abs(b.astype(np.int16) - g.astype(np.int16)) <= 3)
    )

    if not np.any(is_padding):
        return np.zeros((height, width), dtype=np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        is_padding.astype(np.uint8), connectivity=8
    )
    collar_mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 100 and (x == 0 or y == 0 or (x + w) >= width or (y + h) >= height):
            collar_mask[labels == i] = 255

    return collar_mask


def is_optical_marine_image(image: np.ndarray) -> bool:
    """
    Determine whether the input is a natural optical color photograph
    (drone/aerial/RGB satellite) or a SAR radar backscatter scene.
    Optical seawater exhibits Rayleigh scattering and red absorption,
    making Blue and Green substantially higher than Red.
    In SAR (grayscale or enhanced pseudo-RGB), Red (backscatter) >= Blue (texture).
    """
    if image.ndim != 3 or image.shape[2] < 3:
        return False
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    diff_br = float(np.mean(b.astype(np.float32) - r.astype(np.float32)))
    diff_gr = float(np.mean(g.astype(np.float32) - r.astype(np.float32)))
    return diff_br > 20.0 and diff_gr > 10.0


def detect_sar_land_mask(image: np.ndarray) -> np.ndarray:
    """
    Automated SAR Land-Sea Mask.
    Differentiates rugged terrestrial landmasses (coastal mountain ranges,
    urban topography, rocky terrain) from open ocean water based on SAR
    backscatter intensity, Otsu thresholding, and local texture roughness.

    Terrestrial radar shadow valleys (which have dark backscatter identical to
    oil slicks) are engulfed and masked out with the surrounding landmass,
    ensuring that only legitimate maritime oil slicks on the ocean surface are
    segmented and contoured.
    """
    h, w = image.shape[:2]
    scale = 800.0 / max(h, w)
    sh, sw = int(h * scale), int(w * scale)
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

    otsu_th, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Regional standard deviation (texture roughness)
    mean = cv2.blur(gray.astype(np.float32), (15, 15))
    sq_mean = cv2.blur((gray.astype(np.float32)) ** 2, (15, 15))
    local_std = np.sqrt(np.maximum(sq_mean - mean**2, 0))
    mean_scene_std = float(np.mean(local_std))

    # Pure open ocean scenes have low overall texture variance
    if mean_scene_std < 10.0:
        return np.zeros((h, w), dtype=np.uint8)

    # Land seeds: bright reflective ridges, structures, or high backscatter
    land_seed = (gray > max(75.0, otsu_th * 0.9)).astype(np.uint8) * 255
    land_seed = cv2.morphologyEx(
        land_seed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )

    # Morphologically close to unite mountain ridges and engulf shadow valleys
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    closed = cv2.morphologyEx(land_seed, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    solid_land = np.zeros((sh, sw), dtype=np.uint8)
    total_pixels = sh * sw

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        ratio = area / total_pixels
        if ratio > 0.08:  # Significant terrestrial landmass (> 8% of scene)
            comp_mask = labels == i
            comp_roughness = float(np.mean(local_std[comp_mask]))
            if comp_roughness > 15.0:  # Confirmed rugged terrestrial landmass
                solid_land[comp_mask] = 255

    if not np.any(solid_land):
        return np.zeros((h, w), dtype=np.uint8)

    # Fill internal valleys and holes inside the terrestrial landmass
    solid_land = cv2.morphologyEx(
        solid_land,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    )
    cnts, _ = cv2.findContours(solid_land, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        cv2.drawContours(solid_land, [c], -1, 255, thickness=cv2.FILLED)

    # Dilate (15px) to smoothly cover shoreline wave surf and boundaries
    solid_land = cv2.dilate(
        solid_land, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )

    return cv2.resize(solid_land, (w, h), interpolation=cv2.INTER_NEAREST)


def build_roi_mask(height: int, width: int) -> np.ndarray:
    """
    Build a binary ROI mask that excludes the vignette-prone edges of the frame
    (a common source of false positives in raw satellite/aerial captures).
    """
    cfg = config.PREPROCESSING
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    v_margin = cfg.roi_vertical_margin
    h_margin = min(0.02, v_margin * 0.5)
    y_min, y_max = int(height * v_margin), int(height * (1 - v_margin))
    x_min, x_max = int(width * h_margin), int(width * (1 - h_margin))
    roi_mask[y_min:y_max, x_min:x_max] = 255
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
    oil_mask_raw = threshold_oil_regions(hsv, blurred)
    oil_mask_clean = clean_mask_morphology(oil_mask_raw)

    roi_mask = build_roi_mask(height, width)
    oil_mask_clean = cv2.bitwise_and(oil_mask_clean, roi_mask)

    # Filter out synthetic NoData / zero-padding border collars
    collar_mask = detect_nodata_collar(image)
    if np.any(collar_mask):
        oil_mask_clean = cv2.bitwise_and(oil_mask_clean, cv2.bitwise_not(collar_mask))

    # Mask out terrestrial landmasses to prevent radar shadow false positives on SAR imagery.
    # Only applies to SAR imagery where radar shadow valleys mimic oil backscatter;
    # optical marine photos (where water is bright blue/cyan) must not be masked as land.
    land_mask = None
    if not is_optical_marine_image(image):
        land_mask = detect_sar_land_mask(image)
        if np.any(land_mask):
            oil_mask_clean = cv2.bitwise_and(oil_mask_clean, cv2.bitwise_not(land_mask))

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
        land_mask=land_mask,
    )
