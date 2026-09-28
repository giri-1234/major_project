"""
utils.py
--------
Cross-cutting utilities shared by every module in the pipeline:

* Logging setup (console + rotating file handler)
* Filename / extension validation
* Area-estimation and severity/alert classification
* Prediction-history persistence (CSV + JSON)

Keeping these here avoids duplicating logic in prediction.py, routes.py,
mosdac.py, etc.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

from backend import config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """
    Return a module-scoped logger writing to both console and a rotating
    log file at ``config.LOG_FILE``.

    Parameters
    ----------
    name : str
        Usually ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured (avoids duplicate handlers on reload).
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        config.LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# File validation helpers
# ---------------------------------------------------------------------------
def is_allowed_image(filename: str) -> bool:
    """Return True if ``filename`` has an extension permitted by config."""
    _, ext = os.path.splitext(filename.lower())
    return ext in config.ALLOWED_EXTENSIONS


def generate_unique_filename(original_filename: str) -> str:
    """
    Build a collision-free filename that preserves the original extension,
    e.g. 'coast_01.png' -> '20260722_1a2b3c4d_coast_01.png'.
    """
    safe_original = os.path.basename(original_filename).replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_uuid}_{safe_original}"


# ---------------------------------------------------------------------------
# Severity / alert classification
# ---------------------------------------------------------------------------
def classify_severity(area_pct: float) -> str:
    """
    Map an oil-coverage percentage to a severity band.

    Bands (configurable in config.THRESHOLDS):
        area_pct > severity_high_area    -> HIGH
        area_pct > severity_medium_area  -> MEDIUM
        otherwise                        -> LOW
    """
    thresholds = config.THRESHOLDS
    if area_pct > thresholds.severity_high_area:
        return "HIGH"
    if area_pct > thresholds.severity_medium_area:
        return "MEDIUM"
    return "LOW"


def build_alert(status: str, severity: str, area_pct: float) -> Optional[dict]:
    """
    Build an alert payload for a detection result. Returns None when there
    is nothing to alert on (i.e. no spill detected).

    This is intentionally channel-agnostic: today it only populates the
    in-dashboard alert banner (config.ALERTS.enabled_channels = ("dashboard",)).
    To wire email/SMS, extend this function to dispatch through
    smtplib / a Twilio-style client once config.ALERTS.email_enabled /
    sms_enabled are turned on - no other module needs to change.
    """
    if status == "NO OIL SPILL (Look-alike/Clear Water)":
        return None

    return {
        "level": severity,
        "message": f"{severity} severity oil spill detected "
        f"({area_pct:.2f}% surface coverage).",
        "channels_notified": list(config.ALERTS.enabled_channels),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# History persistence (CSV + JSON) — used for the reports / dashboard history
# ---------------------------------------------------------------------------
def calculate_area_km2(area_pct: float, image_source: str = "") -> float:
    """
    Estimate real-world oil spill area in km².

    - Full Sentinel-1 Swath (e.g. raw Copernicus .SAFE archive): ~42,500 km² (250 km × 170 km).
    - Tactical SAR Sector / Sub-scene Crop (e.g. ROI patrol sector, ~16 km × 16 km): ~250.0 km².
    - Manual Coastal / Drone Uploads (local survey): ~10.0 km².

    Parameters
    ----------
    area_pct : float
        Oil coverage as a percentage of the image (0–100).
    image_source : str
        Source tag, e.g. ``"sentinel1:..."`` or ``"manual_upload"``.

    Returns
    -------
    float
        Area in km², rounded to 2 decimal places.
    """
    src = image_source.lower()
    if "full" in src or "raw" in src or "swath" in src:
        scene_area_km2 = 42_500.0  # Full raw Sentinel-1 IW orbital swath
    elif "sentinel1" in src or "sar" in src:
        scene_area_km2 = 250.0     # Tactical sub-scene / patrol sector (~16 km × 16 km)
    else:
        scene_area_km2 = 10.0      # Coastal drone / optical survey
    return round((area_pct / 100.0) * scene_area_km2, 2)


@dataclass
class PredictionRecord:
    """A single row of prediction history, persisted to CSV and JSON."""

    record_id: str
    timestamp: str
    image_source: str
    original_image: str
    stitched_image: Optional[str]
    mask_image: Optional[str]
    status: str
    confidence: float
    area_pct: float
    severity: str
    processing_time_ms: float
    area_km2: float = 0.0
    spill_contours: int = 0


def append_history_record(record: PredictionRecord) -> None:
    """Append a prediction record to both the CSV and JSON history stores."""
    _append_to_csv(record)
    _append_to_json(record)
    logger.info(
        "History recorded: id=%s status=%s severity=%s area=%.2f%%",
        record.record_id,
        record.status,
        record.severity,
        record.area_pct,
    )


def _append_to_csv(record: PredictionRecord) -> None:
    file_exists = os.path.isfile(config.HISTORY_CSV)
    with open(config.HISTORY_CSV, mode="a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(record).keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(record))


def _append_to_json(record: PredictionRecord) -> None:
    history = []
    if os.path.isfile(config.HISTORY_JSON):
        try:
            with open(config.HISTORY_JSON, "r", encoding="utf-8") as fh:
                history = json.load(fh)
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not parse existing history JSON; recreating.")
            history = []

    history.append(asdict(record))

    with open(config.HISTORY_JSON, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)


def read_history(limit: int = 50) -> list:
    """Return the most recent ``limit`` prediction records (newest first)."""
    if not os.path.isfile(config.HISTORY_JSON):
        return []
    try:
        with open(config.HISTORY_JSON, "r", encoding="utf-8") as fh:
            history = json.load(fh)
    except (json.JSONDecodeError, OSError):
        logger.warning("History JSON unreadable; returning empty list.")
        return []
    return list(reversed(history))[:limit]
