"""
satellite/satellite_manager.py
-------------------------------
High-level orchestrator for Sentinel-1 satellite acquisition and preprocessing.

``SatelliteManager.download_latest()`` executes the full workflow:
    1.  Authenticate with CDSE.
    2.  Search for the latest Sentinel-1 GRD product over the Arabian Sea.
    3.  Resolve the product name to its OData UUID.
    4.  Download and extract the product ZIP.
    5.  Copy the raw VV-polarization TIFF to the MOSDAC incoming directory.
    6.  Run SNAP preprocessing (Calibration → Speckle-Filter →
        Terrain-Correction) to produce a georeferenced GeoTIFF.
    7.  Return a structured AcquisitionResult.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field

from backend import config
from backend.satellite.auth import Authenticator
from backend.satellite.download import ProductDownloader
from backend.satellite.preprocessing import PreprocessingError, SARPreprocessor
from backend.satellite.search import ProductSearcher
from backend.utils import get_logger

logger = get_logger(__name__)


@dataclass
class AcquisitionResult:
    """Structured result returned by :meth:`SatelliteManager.download_latest`."""

    product_name: str
    product_id: str
    safe_path: str            # absolute path of extracted SAFE folder
    raw_tiff_path: str        # raw VV TIFF copied to datasets/mosdac/incoming/
    processed_tiff_path: str  # SNAP-processed GeoTIFF in datasets/mosdac/processed_sar/
    preprocessing_skipped: bool = field(default=False)
    preprocessing_error: str = field(default="")

    # Convenience alias kept for backward compatibility with routes.py
    @property
    def tiff_path(self) -> str:
        """Return processed TIFF if available, else raw TIFF."""
        if self.processed_tiff_path and os.path.isfile(self.processed_tiff_path):
            return self.processed_tiff_path
        return self.raw_tiff_path


class SatelliteManager:
    """
    Orchestrates the full Sentinel-1 acquisition + preprocessing pipeline.

    Parameters
    ----------
    incoming_dir : str, optional
        Directory where the raw VV TIFF will be copied.
    extract_dir : str, optional
        Directory where product ZIPs are extracted.
    skip_preprocessing : bool, optional
        If True, SNAP preprocessing is skipped.
    product_id : str, optional
        Specific product name to download. When None, the latest product
        over the Arabian Sea is used (legacy behaviour).
    """

    def __init__(
        self,
        incoming_dir: str | None = None,
        extract_dir: str | None = None,
        skip_preprocessing: bool = False,
        product_id: str | None = None,
        on_progress=None,
    ) -> None:
        self._incoming_dir = incoming_dir or config.SENTINEL1_INCOMING_DIR
        self._extract_dir = extract_dir or config.SATELLITE_EXTRACT_DIR
        self._download_dir = config.SATELLITE_DOWNLOAD_DIR
        self._skip_preprocessing = skip_preprocessing
        self._product_id = product_id
        self._on_progress = on_progress  # callable(step, message) or None

        self._authenticator = Authenticator()
        self._searcher = ProductSearcher()
        self._downloader = ProductDownloader()

    def _progress(
        self,
        step: str,
        message: str,
        percent: float | None = None,
        downloaded_mb: float | None = None,
        total_mb: float | None = None,
    ) -> None:
        """Emit a progress update if a callback is registered."""
        if self._on_progress:
            try:
                self._on_progress(
                    step,
                    message,
                    percent=percent,
                    downloaded_mb=downloaded_mb,
                    total_mb=total_mb,
                )
            except TypeError:
                try:
                    self._on_progress(step, message)
                except Exception:
                    pass
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_latest(self) -> AcquisitionResult:
        """
        Execute the full acquisition + preprocessing workflow.

        Returns
        -------
        AcquisitionResult

        Raises
        ------
        AuthenticationError
            If CDSE credentials are missing or the token endpoint rejects them.
        ProductNotFoundError
            If no Sentinel-1 product is found for the Arabian Sea AOI.
        DownloadError
            If the download or extraction step fails.
        RuntimeError
            If no VV TIFF is found in the extracted .SAFE folder.
        """
        logger.info("Starting satellite acquisition workflow.")

        # 1. Authenticate
        logger.info("Step 1/6 — Authenticating with CDSE.")
        self._progress("authenticating", "Authenticating with Copernicus Data Space…")
        token = self._authenticator.get_token()

        # 2. Find product name — use the user-selected one if provided
        if self._product_id:
            product_name = self._product_id
            logger.info("Step 2/6 — Using user-selected product: %s", product_name)
        else:
            logger.info("Step 2/6 — Searching for latest Sentinel-1 GRD product.")
            self._progress("searching", "Searching for latest Sentinel-1 GRD product…")
            product_name = self._searcher.find_latest_product(token)
            logger.info("Found product: %s", product_name)

        # 3. Resolve to UUID
        logger.info("Step 3/6 — Resolving product UUID for '%s'.", product_name)
        self._progress("searching", f"Resolving product UUID for {product_name[:40]}…")
        product_info = self._searcher.resolve_product_uuid(product_name, token)
        product_id: str = product_info["id"]
        logger.info("Resolved UUID: %s", product_id)

        # 4. Download and extract
        logger.info("Step 4/6 — Downloading and extracting product '%s'.", product_id)
        self._progress("downloading", f"Downloading {product_name[:40]} (this takes 10–30 min)…")

        product_size_mb = 0
        if isinstance(product_info, dict) and product_info.get("size_mb"):
            try:
                product_size_mb = float(product_info["size_mb"])
            except (ValueError, TypeError):
                pass

        def _dl_cb(dl_bytes: int, tot_bytes: int, pct: float):
            dl_mb = round(dl_bytes / (1024 * 1024), 1)
            tot_mb = (
                round(tot_bytes / (1024 * 1024), 1)
                if tot_bytes > 0
                else (product_size_mb or 474.0)
            )
            if tot_bytes == 0 and tot_mb > 0:
                pct = min(99.0, round((dl_mb / tot_mb) * 100, 1))
            msg = f"Downloading satellite data ({pct}% of ~{tot_mb} MB)"
            self._progress(
                "downloading",
                msg,
                percent=pct,
                downloaded_mb=dl_mb,
                total_mb=tot_mb,
            )

        safe_path = self._downloader.download_and_extract(
            product_id=product_id,
            token=token,
            download_dir=self._download_dir,
            extract_dir=self._extract_dir,
            on_progress=_dl_cb,
        )
        logger.info("Extracted .SAFE folder: %s", safe_path)

        # 5. Locate VV TIFF and copy to incoming dir (raw)
        logger.info("Step 5/6 — Staging raw VV TIFF to incoming directory.")
        self._progress("extracting", "Extracting SAFE archive and staging VV TIFF…")
        vv_tiff_path = self._find_vv_tiff(safe_path)
        tiff_filename = os.path.basename(vv_tiff_path)
        raw_dest_path = os.path.join(self._incoming_dir, tiff_filename)
        shutil.copy2(vv_tiff_path, raw_dest_path)

        if not os.path.isfile(raw_dest_path):
            raise RuntimeError(
                f"Copy appeared to succeed but '{raw_dest_path}' does not exist on disk."
            )
        logger.info("Raw VV TIFF staged: %s", raw_dest_path)

        # 6. SNAP preprocessing
        processed_tiff_path = ""
        preprocessing_skipped = False
        preprocessing_error = ""

        if self._skip_preprocessing:
            logger.info("Step 6/6 — SNAP preprocessing skipped (skip_preprocessing=True).")
            preprocessing_skipped = True
        else:
            logger.info("Step 6/6 — Running SNAP preprocessing on .SAFE folder.")
            self._progress("preprocessing", "Running SNAP preprocessing (Calibration → Speckle-Filter → Terrain-Correction)…")
            try:
                preprocessor = SARPreprocessor()
                processed_tiff_path = preprocessor.preprocess(safe_path)
                logger.info("SNAP preprocessing complete: %s", processed_tiff_path)
            except PreprocessingError as exc:
                # Preprocessing failure is non-fatal — raw TIFF is still
                # available. Log as error and surface in the result so the
                # dashboard can tell the user.
                preprocessing_error = str(exc)
                logger.error(
                    "SNAP preprocessing failed (raw TIFF still available): %s", exc
                )

        logger.info(
            "Acquisition workflow complete. product=%s raw_tiff=%s processed_tiff=%s",
            product_name,
            raw_dest_path,
            processed_tiff_path or "(skipped/failed)",
        )

        return AcquisitionResult(
            product_name=product_name,
            product_id=product_id,
            safe_path=os.path.abspath(safe_path),
            raw_tiff_path=os.path.abspath(raw_dest_path),
            processed_tiff_path=os.path.abspath(processed_tiff_path) if processed_tiff_path else "",
            preprocessing_skipped=preprocessing_skipped,
            preprocessing_error=preprocessing_error,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_vv_tiff(self, safe_path: str) -> str:
        """
        Locate the VV-polarization TIFF inside ``<safe_path>/measurement/``.

        Filename must contain ``vv`` (case-insensitive) and end with
        ``.tiff`` or ``.tif``.

        Returns
        -------
        str
            Absolute path to the selected VV TIFF.

        Raises
        ------
        RuntimeError
            If no matching file is found.
        """
        measurement_dir = os.path.join(safe_path, "measurement")

        try:
            entries = os.listdir(measurement_dir)
        except OSError as exc:
            raise RuntimeError(
                f"Could not list measurement directory '{measurement_dir}': {exc}"
            ) from exc

        candidates = sorted(
            entry
            for entry in entries
            if "vv" in entry.lower()
            and entry.lower().endswith((".tiff", ".tif"))
        )

        if not candidates:
            raise RuntimeError(
                f"No VV TIFF found in '{measurement_dir}'. "
                f"Files present: {entries}"
            )

        if len(candidates) > 1:
            logger.warning(
                "Multiple VV TIFFs found in '%s': %s. "
                "Using first in lexicographic order: %s",
                measurement_dir,
                candidates,
                candidates[0],
            )

        return os.path.abspath(os.path.join(measurement_dir, candidates[0]))
