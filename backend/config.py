"""
config.py
---------
Centralized configuration for the Marine Oil Spill Monitoring System.

Every path, threshold, and tunable constant used across the pipeline is
defined here. No other module should hardcode a path or a "magic number" -
they should import from this module instead. This keeps the system easy
to retune (e.g. adjusting detection sensitivity) without touching logic.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------
# PROJECT_ROOT points to the folder that contains backend/, models/, etc.
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")

DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

# Sentinel-1 pipeline directories
SENTINEL1_DIR = os.path.join(DATASETS_DIR, "sentinel1")
SENTINEL1_INCOMING_DIR = os.path.join(SENTINEL1_DIR, "incoming")
SENTINEL1_PROCESSED_DIR = os.path.join(SENTINEL1_DIR, "processed")

# Satellite download / extraction directories
# NOTE: Extract dir uses a short path (C:\s1_extract) to avoid the Windows
# MAX_PATH 260-character limit. The project path is long enough that nested
# .SAFE folder names exceed the limit when extracted inside the project tree.
SATELLITE_DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "downloads")
SATELLITE_EXTRACT_DIR = os.environ.get("SATELLITE_EXTRACT_DIR", r"C:\s1_extract")
SATELLITE_PROCESSED_DIR = os.path.join(SENTINEL1_DIR, "processed_sar")

# Legacy aliases so any remaining references to MOSDAC names still resolve
MOSDAC_DIR = SENTINEL1_DIR
MOSDAC_INCOMING_DIR = SENTINEL1_INCOMING_DIR
MOSDAC_PROCESSED_DIR = SENTINEL1_PROCESSED_DIR

# ---------------------------------------------------------------------------
# SNAP configuration
# ---------------------------------------------------------------------------
# Path to the ESA SNAP gpt command-line tool.
# Override via environment variable SNAP_GPT_PATH if installed elsewhere.
SNAP_GPT_PATH = os.environ.get(
    "SNAP_GPT_PATH",
    r"D:\esa-snap\bin\gpt.exe",
)

# Maximum time in seconds to wait for a single SNAP preprocessing run.
# A full Sentinel-1 IW GRD scene over the Arabian Sea takes ~10-30 min
# depending on hardware. 3600 s (1 hour) is a safe ceiling.
SNAP_TIMEOUT = int(os.environ.get("SNAP_TIMEOUT", "3600"))

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

# Ensure all runtime directories exist at import time.
for _directory in (
    MODELS_DIR,
    UPLOAD_FOLDER,
    SENTINEL1_INCOMING_DIR,
    SENTINEL1_PROCESSED_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    SATELLITE_DOWNLOAD_DIR,
    SATELLITE_EXTRACT_DIR,
    SATELLITE_PROCESSED_DIR,
):
    os.makedirs(_directory, exist_ok=True)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_FILENAME = "oil_spill_attention_model.h5"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)
MODEL_INPUT_SIZE: Tuple[int, int] = (224, 224)  # (width, height) fed to the CNN

# Allowed image extensions for uploads / MOSDAC ingestion / stitching input.
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_UPLOAD_SIZE_MB = 25


# ---------------------------------------------------------------------------
# Preprocessing pipeline constants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessingConfig:
    gaussian_kernel: Tuple[int, int] = (11, 11)
    gaussian_sigma: float = 0.0
    morph_kernel_size: Tuple[int, int] = (5, 5)

    # HSV thresholds tuned for dark oil slicks on water.
    # H = any hue (oil appears in various dark tones including teal/dark-blue)
    # S <= 255 = allows the full saturation range (accommodates dark saturated plumes
    #            in shallow/turquoise optical waters as well as SAR composites)
    # V <= 50  = captures dark pixels characteristic of oil slicks while strictly
    #            excluding clean dark-blue seawater in deep ocean scenes.
    # (Adaptive contrast gating in preprocessing dynamically expands V up to 65
    #  when ambient water is bright turquoise, e.g. tropical shallow waters).
    hsv_lower_oil: Tuple[int, int, int] = (0, 0, 0)
    hsv_upper_oil: Tuple[int, int, int] = (180, 255, 50)

    # Fraction of image height cropped from top/bottom to remove vignette /
    # sensor-edge noise before counting oil pixels.
    roi_vertical_margin: float = 0.05

    # CLAHE (adaptive histogram equalization) parameters for contrast boost.
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)


PREPROCESSING = PreprocessingConfig()


# ---------------------------------------------------------------------------
# Decision / severity thresholds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DetectionThresholds:
    # Primary "confident detection" gate.
    confident_score_min: float = 0.50
    confident_area_min: float = 0.40

    # Secondary "borderline / verify" gate.
    borderline_score_min: float = 0.40
    borderline_area_min: float = 5.0

    # Severity bands, evaluated top-down against oil coverage percentage.
    severity_high_area: float = 20.0
    severity_medium_area: float = 5.0


THRESHOLDS = DetectionThresholds()


# ---------------------------------------------------------------------------
# Alerting configuration (architecture-ready; channels are pluggable)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AlertConfig:
    enabled_channels: Tuple[str, ...] = field(default_factory=lambda: ("dashboard",))
    # Placeholders for future channels - wiring real credentials here is the
    # only change needed to activate them (see mosdac.py / utils.py notes).
    email_enabled: bool = False
    sms_enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    alert_recipients: Tuple[str, ...] = field(default_factory=tuple)


ALERTS = AlertConfig()


# ---------------------------------------------------------------------------
# Image stitching configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StitchingConfig:
    min_match_count: int = 10
    orb_features: int = 4000
    lowe_ratio: float = 0.75
    ransac_reproj_threshold: float = 5.0


STITCHING = StitchingConfig()


# ---------------------------------------------------------------------------
# SAR enhancement (Sentinel-1 preprocessing + RGB composite)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SAREnhancementConfig:
    """
    Controls the Sentinel-1 enhancement pipeline applied after acquisition.

    The pipeline produces three artifacts:
      1. original_gray   — reference stretch only (preserved for comparison)
      2. enhanced_gray   — normalized + speckle-reduced + CLAHE
      3. enhanced_rgb    — 3-channel composite (intensity / gradient / texture)
    """

    enabled: bool = True
    percentile_low: float = 2.0
    percentile_high: float = 98.0
    max_preview_dim: int = 1024

    # CLAHE (adaptive contrast enhancement)
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # Bilateral speckle reduction (edge-preserving; Lee filter can be added later)
    bilateral_d: int = 9
    bilateral_sigma_color: float = 0.08
    bilateral_sigma_space: float = 7.0

    # Local texture window for the blue channel (oil = smooth, water = speckled)
    texture_window: int = 5

    # Which variant feeds the detection model (backward-compatible default)
    model_input_variant: str = "enhanced_rgb"  # enhanced_rgb | enhanced_gray

    # Pluggable stage names — extend SPECKLE_REDUCERS / RGB_BUILDERS in code
    speckle_method: str = "bilateral"
    rgb_method: str = "intensity_gradient_texture"


SAR_ENHANCEMENT = SAREnhancementConfig()

# Backward-compatible alias (older routes referenced SAR_VISUALIZATION)
SAR_VISUALIZATION = SAR_ENHANCEMENT


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = os.path.join(LOGS_DIR, "system.log")
HISTORY_CSV = os.path.join(REPORTS_DIR, "prediction_history.csv")
HISTORY_JSON = os.path.join(REPORTS_DIR, "prediction_history.json")
LOG_LEVEL = os.environ.get("OIL_SPILL_LOG_LEVEL", "INFO")


# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("OIL_SPILL_SECRET_KEY", "dev-key-change-in-production")
DEBUG = os.environ.get("OIL_SPILL_DEBUG", "1") == "1"
HOST = os.environ.get("OIL_SPILL_HOST", "0.0.0.0")
PORT = int(os.environ.get("OIL_SPILL_PORT", "5000"))
