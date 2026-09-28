"""
satellite/sar_enhancement.py
----------------------------
Modular SAR image enhancement and RGB composite pipeline for Sentinel-1.

Design
------
Single-band SAR backscatter is enhanced through a chain of independent,
composable stages.  Each stage is a pure function that can be swapped or
extended without touching the others.

Pipeline stages (in order)
~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **normalize_intensities** — percentile stretch to [0, 1] float32.
2. **reduce_speckle**        — edge-preserving bilateral filter (Lee filter
   can be registered in ``SPECKLE_REDUCERS`` later).
3. **apply_clahe**           — adaptive histogram equalization on the
   denoised image for local contrast boost.
4. **build_rgb_representation** — constructs a meaningful 3-channel BGR
   image from a *single* SAR band by encoding three distinct spatial
   properties:

   ┌─────────┬──────────────────────────────────────────────────────────┐
   │ Channel │ Physical meaning                                         │
   ├─────────┼──────────────────────────────────────────────────────────┤
   │ R (B in BGR) │ CLAHE-enhanced backscatter intensity              │
   │ G         │ Sobel gradient magnitude — oil/water boundaries       │
   │ B         │ Local std-dev — texture (oil smooth, water speckled)  │
   └─────────┴──────────────────────────────────────────────────────────┘

   This is **not** a colormap lookup and **not** grayscale channel
   duplication.  Each channel carries independent information derived
   from the same SAR scene, making the composite suitable for both human
   interpretation and future CNN retraining.

Outputs
-------
Three PNG artifacts are written per scene:

* ``{stem}_sar_original.png``      — reference grayscale (stretch only)
* ``{stem}_sar_enhanced_gray.png`` — speckle-reduced + CLAHE
* ``{stem}_sar_enhanced_rgb.png``  — 3-channel composite for model input
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict

import cv2
import numpy as np

from backend import config
from backend.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SAREnhancementResult:
    """All artifacts produced by the enhancement pipeline."""

    original_gray: np.ndarray       # uint8 H×W — reference only
    enhanced_gray: np.ndarray         # uint8 H×W — denoised + CLAHE
    enhanced_rgb: np.ndarray          # uint8 H×W×3 BGR — composite
    stage_metadata: dict = field(default_factory=dict)


@dataclass
class SARFileOutputs:
    """Filenames (basenames) written to the upload folder."""

    original_gray: str
    enhanced_gray: str
    enhanced_rgb: str
    model_input: str                  # whichever variant feeds the detector


# ---------------------------------------------------------------------------
# Stage 1 — Intensity normalization
# ---------------------------------------------------------------------------


def normalize_intensities(
    raw: np.ndarray,
    percentile_low: float | None = None,
    percentile_high: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Percentile-stretch raw backscatter to [0, 1] float32.

    Returns
    -------
    normalized : float32 H×W in [0, 1]
    reference_uint8 : uint8 H×W — stretch-only reference image
    """
    cfg = config.SAR_ENHANCEMENT
    p_low = cfg.percentile_low if percentile_low is None else percentile_low
    p_high = cfg.percentile_high if percentile_high is None else percentile_high

    image = raw.astype(np.float32)
    valid = image[image > 0]
    if valid.size > 0:
        lo = float(np.percentile(valid, p_low))
        hi = float(np.percentile(valid, p_high))
    else:
        lo = float(image.min())
        hi = float(image.max())
        if hi <= lo:
            hi = lo + 1.0

    clipped = np.clip(image, lo, hi)
    normalized = (clipped - lo) / (hi - lo + 1e-10)
    reference_uint8 = (normalized * 255).astype(np.uint8)
    return normalized, reference_uint8


def _to_uint8(channel: np.ndarray) -> np.ndarray:
    """Percentile-stretch a single channel array to uint8."""
    arr = channel.astype(np.float32)
    valid = arr[arr > 0] if np.any(arr > 0) else arr.ravel()
    if valid.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(valid, 2))
    hi = float(np.percentile(valid, 98))
    stretched = np.clip(arr, lo, hi)
    return ((stretched - lo) / (hi - lo + 1e-10) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stage 2 — Speckle reduction (pluggable)
# ---------------------------------------------------------------------------


def reduce_speckle_bilateral(normalized: np.ndarray) -> np.ndarray:
    """
    Edge-preserving bilateral filter on float32 [0, 1] SAR data.

    Bilateral filtering reduces multiplicative speckle while preserving
    slick boundaries — a practical alternative to a full Lee filter that
    keeps the dependency footprint small (OpenCV only).
    """
    cfg = config.SAR_ENHANCEMENT
    # OpenCV bilateral expects uint8 or float; sigmaColor scales with range
    return cv2.bilateralFilter(
        normalized,
        d=cfg.bilateral_d,
        sigmaColor=cfg.bilateral_sigma_color,
        sigmaSpace=cfg.bilateral_sigma_space,
    )


SPECKLE_REDUCERS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "bilateral": reduce_speckle_bilateral,
}


def reduce_speckle(normalized: np.ndarray, method: str | None = None) -> np.ndarray:
    """Dispatch to a registered speckle-reduction method."""
    cfg = config.SAR_ENHANCEMENT
    name = (method or cfg.speckle_method).lower()
    fn = SPECKLE_REDUCERS.get(name)
    if fn is None:
        logger.warning("Unknown speckle method '%s'; using bilateral.", name)
        fn = reduce_speckle_bilateral
    return fn(normalized)


# ---------------------------------------------------------------------------
# Stage 3 — Adaptive contrast (CLAHE)
# ---------------------------------------------------------------------------


def apply_clahe(gray_uint8: np.ndarray) -> np.ndarray:
    """Apply CLAHE to a single-channel uint8 image."""
    cfg = config.SAR_ENHANCEMENT
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=cfg.clahe_tile_grid_size,
    )
    return clahe.apply(gray_uint8)


# ---------------------------------------------------------------------------
# Stage 4 — Meaningful RGB composite (pluggable)
# ---------------------------------------------------------------------------


def _local_std_dev(image: np.ndarray, window: int) -> np.ndarray:
    """Compute local standard deviation using a box filter (texture map)."""
    img = image.astype(np.float32)
    k = max(3, window | 1)  # ensure odd kernel size
    mean = cv2.blur(img, (k, k))
    mean_sq = cv2.blur(img * img, (k, k))
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(variance)


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude — highlights oil/water boundaries."""
    img = image.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def build_rgb_intensity_gradient_texture(
    enhanced_gray: np.ndarray,
    denoised: np.ndarray,
) -> np.ndarray:
    """
    Build a 3-channel BGR composite from single-band SAR data.

    Each channel encodes a distinct spatial property of the scene:

    * **R** — CLAHE-enhanced backscatter (primary intensity).
    * **G** — Gradient magnitude on the denoised image (structure/edges).
    * **B** — Local texture variance (oil slicks appear smoother than water).

    Channels are independently normalized to uint8 before merging.
    """
    cfg = config.SAR_ENHANCEMENT

    r_channel = enhanced_gray
    g_channel = _to_uint8(_gradient_magnitude(denoised))
    b_channel = _to_uint8(_local_std_dev(denoised, cfg.texture_window))

    return cv2.merge([b_channel, g_channel, r_channel])  # BGR order


RGB_BUILDERS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "intensity_gradient_texture": build_rgb_intensity_gradient_texture,
}


def build_rgb_representation(
    enhanced_gray: np.ndarray,
    denoised: np.ndarray,
    method: str | None = None,
) -> np.ndarray:
    """Dispatch to a registered RGB composite builder."""
    cfg = config.SAR_ENHANCEMENT
    name = (method or cfg.rgb_method).lower()
    fn = RGB_BUILDERS.get(name)
    if fn is None:
        logger.warning("Unknown RGB method '%s'; using intensity_gradient_texture.", name)
        fn = build_rgb_intensity_gradient_texture
    return fn(enhanced_gray, denoised)


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------


def run_sar_enhancement_pipeline(raw: np.ndarray) -> SAREnhancementResult:
    """
    Execute all enhancement stages on a raw SAR array.

    Parameters
    ----------
    raw : np.ndarray
        Raw backscatter (float32 or integer, H×W).

    Returns
    -------
    SAREnhancementResult
    """
    cfg = config.SAR_ENHANCEMENT

    # Stage 1 — normalize + preserve reference
    normalized, original_gray = normalize_intensities(raw)

    # Stage 2 — speckle reduction
    denoised = reduce_speckle(normalized, method=cfg.speckle_method)

    # Stage 3 — CLAHE on denoised image
    denoised_uint8 = (np.clip(denoised, 0, 1) * 255).astype(np.uint8)
    enhanced_gray = apply_clahe(denoised_uint8)

    # Stage 4 — meaningful RGB composite
    enhanced_rgb = build_rgb_representation(
        enhanced_gray, denoised, method=cfg.rgb_method
    )

    metadata = {
        "speckle_method": cfg.speckle_method,
        "rgb_method": cfg.rgb_method,
        "shape": raw.shape,
    }

    logger.debug(
        "SAR enhancement complete: shape=%s speckle=%s rgb=%s",
        raw.shape,
        cfg.speckle_method,
        cfg.rgb_method,
    )

    return SAREnhancementResult(
        original_gray=original_gray,
        enhanced_gray=enhanced_gray,
        enhanced_rgb=enhanced_rgb,
        stage_metadata=metadata,
    )


def _load_raw_array(filepath: str, max_dim: int) -> np.ndarray:
    """Load raw SAR data from GeoTIFF or standard image file."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".tif", ".tiff"):
        from backend.satellite.cog_reader import read_cog_overview

        return read_cog_overview(filepath, max_dim=max_dim, normalize=False)

    raw = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Could not read SAR image: {filepath}")
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    return raw.astype(np.float32)


def process_sar_scene(
    input_path: str,
    output_dir: str,
    *,
    force: bool = False,
) -> SARFileOutputs:
    """
    Run the full SAR enhancement pipeline and write three PNG artifacts.

    Parameters
    ----------
    input_path : str
        Path to the incoming GeoTIFF or grayscale image.
    output_dir : str
        Directory for output PNGs (typically ``static/uploads/``).
    force : bool
        Re-process even if output files already exist.

    Returns
    -------
    SARFileOutputs
        Basenames of the three written files plus the model-input variant.
    """
    cfg = config.SAR_ENHANCEMENT
    stem = os.path.splitext(os.path.basename(input_path))[0]

    names = SARFileOutputs(
        original_gray=f"{stem}_sar_original.png",
        enhanced_gray=f"{stem}_sar_enhanced_gray.png",
        enhanced_rgb=f"{stem}_sar_enhanced_rgb.png",
        model_input="",
    )

    paths = {
        "original": os.path.join(output_dir, names.original_gray),
        "enhanced_gray": os.path.join(output_dir, names.enhanced_gray),
        "enhanced_rgb": os.path.join(output_dir, names.enhanced_rgb),
    }

    if (
        not force
        and all(os.path.isfile(p) for p in paths.values())
    ):
        logger.info("SAR enhancement outputs already exist for '%s'; skipping.", stem)
    else:
        raw = _load_raw_array(input_path, max_dim=cfg.max_preview_dim)
        if max(raw.shape) > cfg.max_preview_dim:
            scale = cfg.max_preview_dim / max(raw.shape)
            raw = cv2.resize(
                raw,
                (int(raw.shape[1] * scale), int(raw.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        result = run_sar_enhancement_pipeline(raw)

        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(paths["original"], result.original_gray)
        cv2.imwrite(paths["enhanced_gray"], result.enhanced_gray)
        cv2.imwrite(paths["enhanced_rgb"], result.enhanced_rgb)
        logger.info(
            "SAR enhancement saved: original=%s enhanced_gray=%s enhanced_rgb=%s",
            names.original_gray,
            names.enhanced_gray,
            names.enhanced_rgb,
        )

    variant = cfg.model_input_variant.lower()
    if variant == "enhanced_gray":
        names.model_input = names.enhanced_gray
    else:
        names.model_input = names.enhanced_rgb

    return names


def get_model_input_path(outputs: SARFileOutputs, output_dir: str) -> str:
    """Return the absolute path to the variant used for detection."""
    return os.path.join(output_dir, outputs.model_input)


# ---------------------------------------------------------------------------
# Image statistics
# ---------------------------------------------------------------------------


def compute_image_statistics(image: np.ndarray) -> dict:
    """
    Compute scientific image statistics for a SAR scene or any grayscale /
    single-channel image.

    Parameters
    ----------
    image : np.ndarray
        uint8 grayscale or BGR image.  If BGR (3-channel), it is converted
        to grayscale before computing statistics so all metrics share a
        common single-channel basis.

    Returns
    -------
    dict with keys:
        mean          – arithmetic mean pixel value (0–255)
        std_dev       – population standard deviation
        contrast      – RMS contrast  (std_dev / mean, or 0 when mean == 0)
        dynamic_range – difference between 98th- and 2nd-percentile pixel
                        values (more robust than max − min for noisy SAR data)
        entropy       – Shannon entropy in bits (uses histogram of 256 bins)
    """
    # Reduce to single channel
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = gray.astype(np.float32)
    flat = gray.ravel()

    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    contrast = round(std_val / mean_val, 4) if mean_val > 0 else 0.0

    # Robust dynamic range using percentiles
    p2 = float(np.percentile(flat, 2))
    p98 = float(np.percentile(flat, 98))
    dynamic_range = round(p98 - p2, 2)

    # Shannon entropy from normalised histogram
    hist, _ = np.histogram(flat, bins=256, range=(0, 255))
    hist_nonzero = hist[hist > 0].astype(np.float64)
    prob = hist_nonzero / hist_nonzero.sum()
    entropy = float(-np.sum(prob * np.log2(prob)))

    return {
        "mean": round(mean_val, 2),
        "std_dev": round(std_val, 2),
        "contrast": contrast,
        "dynamic_range": dynamic_range,
        "entropy": round(entropy, 3),
    }
