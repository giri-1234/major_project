"""
prediction.py
-------------
Deep-learning inference and decision logic for the oil spill detector.

Preserves the exact decision gates from the original single-file app.py
(confidence + area double-gate, severity bands) but wraps them in a
reusable, testable ``OilSpillPredictor`` class and exposes structured
results instead of magic-tuple returns.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from backend import config
from backend.preprocessing import PreprocessedResult, run_preprocessing_pipeline
from backend.utils import calculate_area_km2, classify_severity, get_logger

# Raise Pillow's decompression-bomb limit for large satellite GeoTIFFs
Image.MAX_IMAGE_PIXELS = None

logger = get_logger(__name__)

# TensorFlow is imported lazily inside the class so that modules which only
# need preprocessing/stitching (e.g. quick unit tests) don't pay the import
# cost or require TensorFlow to be installed.
_tf = None


def _lazy_import_tensorflow():
    global _tf
    if _tf is None:
        import tensorflow as tf  # noqa: WPS433 (intentional lazy import)

        _tf = tf
    return _tf


@dataclass
class PredictionResult:
    """Structured output of a full detect-and-segment run on one image."""

    status: str
    color: str
    confidence: float
    area_pct: float
    severity: str
    mask_filename: Optional[str]
    processing_time_ms: float
    area_km2: float = 0.0
    tile_count: int = 0
    contour_filename: Optional[str] = None
    spill_contours: int = 0


class OilSpillPredictor:
    """
    Loads the trained attention-based MobileNet model once and exposes a
    single ``predict`` entry point that mirrors the original project's
    ``pro_analysis`` function, decomposed into clear stages.
    """

    _instance: Optional["OilSpillPredictor"] = None

    def __init__(self, model_path: str = config.MODEL_PATH):
        tf = _lazy_import_tensorflow()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. Place the trained "
                f"'.h5' model there or update config.MODEL_PATH."
            )
        logger.info("Loading model from %s ...", model_path)
        self._model = tf.keras.models.load_model(model_path)
        self._tf = tf
        logger.info("Model loaded successfully.")

    @classmethod
    def get_instance(cls) -> "OilSpillPredictor":
        """Simple singleton accessor so the (heavy) model loads only once."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Inference stages
    # ------------------------------------------------------------------
    def _predict_raw_score(self, filepath: str) -> float:
        """Run the CNN classifier and return its raw sigmoid confidence score."""
        img_load = self._tf.keras.utils.load_img(
            filepath, target_size=config.MODEL_INPUT_SIZE
        )
        x = self._tf.keras.utils.img_to_array(img_load)
        x = np.expand_dims(x, axis=0)
        raw_score = self._model.predict(x, verbose=0)[0][0]
        return float(raw_score)

    @staticmethod
    def _estimate_area(preprocessed: PreprocessedResult) -> float:
        """Compute the percentage of ROI-masked pixels classified as oil."""
        mask = preprocessed.oil_mask_clean
        height, width = mask.shape[:2]
        oil_pixels = cv2.countNonZero(mask)
        total_pixels = height * width
        return (oil_pixels / total_pixels) * 100 if total_pixels else 0.0

    @staticmethod
    def _decide(raw_score: float, area_pct: float) -> tuple:
        """
        Apply the double-gate decision logic (confidence AND area) and
        assign a severity band. Thresholds come from config.THRESHOLDS so
        they can be retuned without touching this logic.
        """
        t = config.THRESHOLDS

        if raw_score > t.confident_score_min and area_pct > t.confident_area_min:
            status = "OIL SPILL DETECTED"
            color = "red"
            severity = classify_severity(area_pct)
            return status, color, severity, area_pct

        if raw_score > t.borderline_score_min and area_pct > t.borderline_area_min:
            status = "OIL SPILL DETECTED (Borderline/Verify)"
            color = "orange"
            severity = "LOW"
            return status, color, severity, area_pct

        status = "NO OIL SPILL (Look-alike/Clear Water)"
        color = "green"
        return status, color, "N/A", 0.0

    def _save_mask(self, filepath: str, mask: np.ndarray) -> str:
        """Save the binary segmentation mask alongside the uploaded image."""
        mask_filename = "mask_" + os.path.basename(filepath)
        mask_path = os.path.join(config.UPLOAD_FOLDER, mask_filename)
        cv2.imwrite(mask_path, mask)
        return mask_filename

    def _save_contour_overlay(
        self, filepath: str, original_image: np.ndarray, mask: np.ndarray
    ) -> str:
        """Draw oil spill contours on the original image and save as overlay."""
        overlay = original_image.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter small noise contours (scaled by resolution)
        h, w = mask.shape[:2]
        min_area = 25 if min(h, w) < 400 else 50
        significant = [c for c in contours if cv2.contourArea(c) > min_area]

        # Draw filled semi-transparent overlay
        oil_overlay = overlay.copy()
        cv2.fillPoly(oil_overlay, significant, (0, 0, 180))  # dark red fill
        overlay = cv2.addWeighted(overlay, 0.7, oil_overlay, 0.3, 0)

        # Draw contour borders in bright cyan
        cv2.drawContours(overlay, significant, -1, (55, 224, 196), 2)

        contour_filename = "contour_" + os.path.basename(filepath)
        contour_path = os.path.join(config.UPLOAD_FOLDER, contour_filename)
        cv2.imwrite(contour_path, overlay)
        return contour_filename

    @staticmethod
    def _count_contours(mask: np.ndarray) -> int:
        """Return the number of significant oil-spill regions in a binary mask."""
        h, w = mask.shape[:2]
        min_area = 25 if min(h, w) < 400 else 50
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return len([c for c in contours if cv2.contourArea(c) > min_area])

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def predict(
        self, filepath: str, image_source: str = "", tile_count: int = 0
    ) -> PredictionResult:
        """
        Run the complete detect -> segment -> estimate-area -> decide
        pipeline on a single image file and return a ``PredictionResult``.

        Parameters
        ----------
        filepath : str
            Path to the image file to analyse.
        image_source : str
            Source tag used for area estimation (e.g. ``"sentinel1:..."``,
            ``"manual_upload"``).
        tile_count : int
            Number of tiling-pipeline tiles processed upstream (0 if no
            tiling was performed).
        """
        start_time = time.perf_counter()

        raw_score = self._predict_raw_score(filepath)

        # Load image with downscaling for large SAR GeoTIFFs
        image = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {filepath}")

        # Downscale large SAR scenes (Sentinel-1 can be ~15000x15000)
        h, w = image.shape[:2]
        if max(h, w) > 4096:
            scale = 4096 / max(h, w)
            image = cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        # Convert float32 SAR data to uint8 for the preprocessing pipeline
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            p2 = float(np.percentile(image[image > 0], 2)) if np.any(image > 0) else 0.0
            p98 = (
                float(np.percentile(image[image > 0], 98))
                if np.any(image > 0)
                else 1.0
            )
            image = np.clip(image, p2, p98)
            image = ((image - p2) / (p98 - p2 + 1e-10) * 255).astype(np.uint8)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)

        preprocessed = run_preprocessing_pipeline(image)
        area_pct = self._estimate_area(preprocessed)

        status, color, severity, reported_area = self._decide(raw_score, area_pct)

        mask = preprocessed.oil_mask_clean
        mask_filename = None
        contour_filename = None
        spill_contours = 0

        if status != "NO OIL SPILL (Look-alike/Clear Water)":
            mask_filename = self._save_mask(filepath, mask)
            spill_contours = self._count_contours(mask)
            try:
                contour_filename = self._save_contour_overlay(filepath, image, mask)
            except Exception:  # noqa: BLE001
                logger.warning("Could not save contour overlay for %s", filepath)
        else:
            # When decision is NO OIL SPILL, do not report false noise contours or draw red/cyan spill markings
            spill_contours = 0
            try:
                contour_filename = self._save_contour_overlay(
                    filepath, image, np.zeros_like(mask)
                )
            except Exception:  # noqa: BLE001
                logger.warning("Could not save contour overlay for %s", filepath)

        area_km2 = calculate_area_km2(reported_area, image_source)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Prediction complete: file=%s status=%s score=%.3f area=%.2f%% "
            "severity=%s contours=%d area_km2=%.2f time=%.1fms",
            os.path.basename(filepath),
            status,
            raw_score,
            reported_area,
            severity,
            spill_contours,
            area_km2,
            elapsed_ms,
        )

        return PredictionResult(
            status=status,
            color=color,
            confidence=raw_score,
            area_pct=round(reported_area, 2),
            severity=severity,
            mask_filename=mask_filename,
            processing_time_ms=round(elapsed_ms, 1),
            area_km2=area_km2,
            tile_count=tile_count,
            contour_filename=contour_filename,
            spill_contours=spill_contours,
        )
