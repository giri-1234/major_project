"""
satellite/preprocessing.py
---------------------------
SNAP-based Sentinel-1 GRD preprocessing module.

Responsibility: given a `.SAFE` folder path, run the standard SAR preprocessing
chain using ESA SNAP's ``gpt`` command-line tool:

    Read → Calibration (Sigma0 VV) → Speckle-Filter (Lee 5x5)
         → Terrain-Correction (UTM 10m) → Write (GeoTIFF)

The output is a Float32 GeoTIFF ready for model inference or dataset
construction.  No rasterio dependency is required — SNAP handles all I/O.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from backend import config
from backend.utils import get_logger

logger = get_logger(__name__)

# Path to the SNAP graph XML shipped with this package
_GRAPH_FILE = os.path.join(
    os.path.dirname(__file__), "graphs", "s1_grd_preprocessing.xml"
)


class PreprocessingError(Exception):
    """Raised when the SNAP gpt preprocessing step fails."""


class SARPreprocessor:
    """
    Runs the Sentinel-1 GRD preprocessing chain via ESA SNAP ``gpt``.

    Parameters
    ----------
    gpt_path : str, optional
        Absolute path to the ``gpt`` executable.
    output_dir : str, optional
        Directory where the processed GeoTIFF will be written.
    timeout : int, optional
        Maximum seconds to wait for SNAP to finish (default 3600).
    """

    def __init__(
        self,
        gpt_path: str | None = None,
        output_dir: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self._gpt_path = gpt_path or config.SNAP_GPT_PATH
        self._output_dir = output_dir or config.SATELLITE_PROCESSED_DIR
        self._timeout = timeout or config.SNAP_TIMEOUT
        self._graph_file = _GRAPH_FILE

        if not os.path.isfile(self._gpt_path):
            raise PreprocessingError(
                f"SNAP gpt executable not found at '{self._gpt_path}'. "
                "Update SNAP_GPT_PATH in config.py."
            )

        os.makedirs(self._output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, safe_path: str) -> str:
        """
        Run the full preprocessing chain on a Sentinel-1 .SAFE product.

        For COG (Cloud Optimized GeoTIFF) products — identified by '_COG' in
        the folder name — SNAP preprocessing is skipped because Copernicus has
        already applied radiometric calibration and terrain correction. The VV
        TIFF is used directly.

        For standard GRD products, runs:
          Read → Calibration (Sigma0 VV) → Speckle-Filter (Lee 5x5)
               → Terrain-Correction (SRTM 3Sec, UTM 10m) → GeoTIFF

        Returns
        -------
        str
            Absolute path to the output GeoTIFF.

        Raises
        ------
        PreprocessingError
            If SNAP fails or output file is not produced.
        FileNotFoundError
            If safe_path does not exist.
        """
        safe_path = os.path.abspath(safe_path)

        if not os.path.isdir(safe_path):
            raise FileNotFoundError(f"SAFE folder not found: '{safe_path}'")

        product_stem = Path(safe_path).stem

        # COG products are already calibrated + terrain-corrected by Copernicus
        # No SNAP processing needed — copy the VV TIFF directly to output dir
        if "_COG" in safe_path.upper():
            logger.info(
                "COG product detected — skipping SNAP (already preprocessed by Copernicus)."
            )
            return self._handle_cog_product(safe_path, product_stem)

        # Standard GRD — run full SNAP chain
        output_tiff = os.path.join(self._output_dir, f"{product_stem}_processed.tif")
        logger.info(
            "Starting SNAP preprocessing: '%s' → '%s'",
            os.path.basename(safe_path),
            output_tiff,
        )
        start_time = time.time()
        self._run_gpt(input_path=safe_path, output_path=output_tiff)
        elapsed = time.time() - start_time

        if not os.path.isfile(output_tiff):
            raise PreprocessingError(
                f"SNAP completed but output file not found: '{output_tiff}'"
            )

        size_mb = os.path.getsize(output_tiff) / (1024 * 1024)
        logger.info(
            "Preprocessing complete in %.1f s — %s (%.1f MB)",
            elapsed, output_tiff, size_mb,
        )
        return output_tiff

    def _handle_cog_product(self, safe_path: str, product_stem: str) -> str:
        """
        For COG products, locate the VV TIFF and copy it to the output dir.
        COG products from Copernicus are already radiometrically calibrated
        and terrain-corrected — no further SNAP processing is needed.
        """
        measurement_dir = os.path.join(safe_path, "measurement")
        if not os.path.isdir(measurement_dir):
            raise PreprocessingError(
                f"measurement/ folder not found in COG product: '{safe_path}'"
            )

        # Find VV TIFF
        candidates = sorted(
            f for f in os.listdir(measurement_dir)
            if "vv" in f.lower() and f.lower().endswith((".tiff", ".tif"))
        )
        if not candidates:
            raise PreprocessingError(
                f"No VV TIFF found in '{measurement_dir}'. "
                f"Files: {os.listdir(measurement_dir)}"
            )

        src = os.path.join(measurement_dir, candidates[0])
        dest = os.path.join(self._output_dir, f"{product_stem}_vv_cog.tif")

        logger.info("Copying COG VV TIFF: %s → %s", src, dest)
        shutil.copy2(src, dest)

        size_mb = os.path.getsize(dest) / (1024 * 1024)
        logger.info("COG product ready: %s (%.1f MB)", dest, size_mb)
        return dest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_gpt(self, input_path: str, output_path: str) -> None:
        """
        Execute SNAP gpt with the preprocessing graph.

        The graph XML is copied to a short temp path first to avoid the
        Windows MAX_PATH (260-char) limit that causes SNAP to fail when the
        project directory is deeply nested.
        """
        # Copy graph XML to a short temp path
        tmp_graph = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".xml", prefix="s1_graph_", delete=False,
                dir=os.path.expanduser("~")
            ) as tmp:
                tmp_graph = tmp.name
            shutil.copy2(self._graph_file, tmp_graph)

            cmd = [
                self._gpt_path,
                tmp_graph,
                f"-Pinput={input_path}",
                f"-Poutput={output_path}",
                "-c", "4G",
                "-q", "4",
            ]

            logger.info("Running: %s", " ".join(cmd))

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                raise PreprocessingError(
                    f"Failed to launch gpt at '{self._gpt_path}': {exc}"
                ) from exc

            output_lines: list[str] = []
            try:
                for line in process.stdout:  # type: ignore[union-attr]
                    line = line.rstrip()
                    if line:
                        output_lines.append(line)
                        if "%" in line or "done" in line.lower() or "error" in line.lower():
                            logger.info("[SNAP] %s", line)
                        else:
                            logger.debug("[SNAP] %s", line)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error reading gpt output: %s", exc)

            try:
                process.wait(timeout=self._timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                raise PreprocessingError(
                    f"SNAP gpt timed out after {self._timeout}s."
                )

            if process.returncode != 0:
                tail = "\n".join(output_lines[-20:])
                raise PreprocessingError(
                    f"SNAP gpt exited with code {process.returncode} "
                    f"processing '{os.path.basename(input_path)}'.\n"
                    f"Last output:\n{tail}"
                )
        finally:
            if tmp_graph and os.path.exists(tmp_graph):
                try:
                    os.unlink(tmp_graph)
                except OSError:
                    pass
