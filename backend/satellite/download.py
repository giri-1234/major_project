"""
satellite/download.py
---------------------
CDSE product download and extraction module.

Responsibility: stream a product ZIP from the CDSE Download Endpoint, write it
to disk in chunks, extract the archive using stdlib ``zipfile``, and return the
absolute path of the resulting ``.SAFE`` folder.
"""

from __future__ import annotations

import os
import shutil
import zipfile

import requests

from backend.satellite import DownloadError

_DOWNLOAD_URL_TEMPLATE = (
    "https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
)

_TIMEOUT = 300          # 300 s to accommodate ~474 MB products
_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB


class ProductDownloader:
    """Downloads and extracts a CDSE Sentinel-1 product ZIP."""

    def download_and_extract(
        self,
        product_id: str,
        token: str,
        download_dir: str,
        extract_dir: str,
    ) -> str:
        """
        Stream-download a product ZIP, extract it, and return the ``.SAFE``
        folder path.

        Parameters
        ----------
        product_id : str
            CDSE OData UUID of the product to download.
        token : str
            A valid CDSE Bearer access token.
        download_dir : str
            Directory in which the ZIP file will be written.
        extract_dir : str
            Directory to which the ZIP will be extracted.

        Returns
        -------
        str
            Absolute path to the extracted ``.SAFE`` folder.

        Raises
        ------
        DownloadError
            If the download endpoint returns non-2xx, or if no ``.SAFE`` folder
            is found after extraction.
        """
        url = _DOWNLOAD_URL_TEMPLATE.format(product_id=product_id)
        zip_path = os.path.join(download_dir, f"{product_id}.zip")

        headers = {"Authorization": f"Bearer {token}"}

        # --- Download (streaming, 1 MB chunks) ---
        try:
            response = requests.get(
                url, headers=headers, timeout=_TIMEOUT, stream=True
            )
        except requests.RequestException as exc:
            raise DownloadError(
                f"HTTP request for product '{product_id}' failed: {exc}"
            ) from exc

        if not response.ok:
            raise DownloadError(
                f"Download endpoint returned HTTP {response.status_code} "
                f"for product UUID '{product_id}'."
            )

        with open(zip_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    fh.write(chunk)

        # --- Extract (entry-by-entry to handle Windows path issues) ---
        self._safe_extract(zip_path, extract_dir)

        # --- Locate .SAFE folder ---
        safe_path = self._find_safe_folder(extract_dir, product_id)
        return safe_path

    @staticmethod
    def _safe_extract(zip_path: str, extract_dir: str) -> None:
        """
        Extract a ZIP file entry-by-entry with explicit directory creation.

        Uses split-based path joining to avoid Windows mixed-separator issues
        and ensures parent directories exist before writing each file.
        This handles the Windows MAX_PATH issue when extract_dir is a short path.
        """
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                # Split on forward slash (ZIP spec) to get clean path parts
                parts = member.filename.replace("\\", "/").split("/")
                target_path = os.path.join(extract_dir, *parts)

                # Always create parent directories
                parent = os.path.dirname(target_path)
                os.makedirs(parent, exist_ok=True)

                # Skip directory entries
                if member.filename.endswith("/"):
                    continue

                # Write the file
                with zf.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)

    @staticmethod
    def _find_safe_folder(extract_dir: str, product_id: str) -> str:
        """
        Scan ``extract_dir`` (one level deep) for an entry ending in ``.SAFE``
        and return its absolute path.

        Raises
        ------
        DownloadError
            If no ``.SAFE`` folder is found.
        """
        found = []
        try:
            entries = os.listdir(extract_dir)
        except OSError as exc:
            raise DownloadError(
                f"Could not list extraction directory '{extract_dir}': {exc}"
            ) from exc

        for entry in entries:
            if entry.upper().endswith(".SAFE"):
                full_path = os.path.join(extract_dir, entry)
                if os.path.isdir(full_path):
                    found.append(os.path.abspath(full_path))

        if not found:
            raise DownloadError(
                f"No .SAFE folder found in '{extract_dir}' after extracting "
                f"product '{product_id}'. "
                f"Entries found: {os.listdir(extract_dir)}"
            )

        # Return the first in lexicographic order for determinism.
        return sorted(found)[0]
