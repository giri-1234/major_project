"""
mosdac.py
---------
Satellite image acquisition from MOSDAC (Meteorological & Oceanographic
Satellite Data Archival Centre).

IMPORTANT - honesty note:
This module does NOT call a live MOSDAC API, because doing so would
require registered credentials this project does not have. Instead it
implements a **folder-based ingestion** strategy: operators (or a cron
job / future API client) drop satellite tiles into
``datasets/mosdac/incoming/``, and ``MosdacIngestor`` picks them up,
groups them into a "scene" (one acquisition = one or more tiles), and
hands them to the stitching/prediction pipeline. Processed tiles are
moved to ``datasets/mosdac/processed/`` so they aren't re-ingested.

Swapping in the real API later only requires implementing
``fetch_from_api()`` below - every other module (routes.py, stitching.py,
prediction.py) is agnostic to where the tiles came from.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import List

from backend import config
from backend.utils import get_logger, is_allowed_image

logger = get_logger(__name__)


@dataclass
class MosdacScene:
    """One "acquisition" grouping - one or more raw tiles ready to stitch."""

    scene_id: str
    tile_paths: List[str]
    acquired_at: str


class MosdacIngestor:
    """
    Watches the local MOSDAC drop folder and surfaces new scenes to the
    processing pipeline.

    Future extension point
    -----------------------
    To connect a real MOSDAC API (or any other satellite data provider),
    implement ``fetch_from_api()`` to download tiles into
    ``config.MOSDAC_INCOMING_DIR`` (or return paths directly), and call
    it from ``fetch_new_images()`` before the folder scan. No other part
    of the system needs to change.
    """

    def __init__(
        self,
        incoming_dir: str = config.MOSDAC_INCOMING_DIR,
        processed_dir: str = config.MOSDAC_PROCESSED_DIR,
    ):
        self.incoming_dir = incoming_dir
        self.processed_dir = processed_dir
        os.makedirs(self.incoming_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Current strategy: local folder ingestion
    # ------------------------------------------------------------------
    def scan_incoming_folder(self) -> List[str]:
        """Return full paths of all valid, unprocessed images waiting in the drop folder."""
        candidates = sorted(os.listdir(self.incoming_dir))
        image_paths = [
            os.path.join(self.incoming_dir, name)
            for name in candidates
            if is_allowed_image(name)
            and os.path.isfile(os.path.join(self.incoming_dir, name))
        ]
        logger.info("MOSDAC folder scan found %d image(s).", len(image_paths))
        return image_paths

    def fetch_new_images(self):
        """
        Build a single ``MosdacScene`` from everything currently sitting in
        the incoming folder. Raises if nothing is available.
        """
        tile_paths = self.scan_incoming_folder()
        if not tile_paths:
            raise FileNotFoundError(
                f"No new images found in '{self.incoming_dir}'. Drop satellite "
                f"tiles there (PNG/JPEG/GeoTIFF) and try again."
            )

        scene_id = datetime.now().strftime("MOSDAC_%Y%m%d_%H%M%S")
        return MosdacScene(
            scene_id=scene_id,
            tile_paths=tile_paths,
            acquired_at=datetime.utcnow().isoformat() + "Z",
        )

    def mark_processed(self, scene: MosdacScene) -> None:
        """Move a scene's tiles out of the incoming folder once processed."""
        for tile_path in scene.tile_paths:
            if os.path.isfile(tile_path):
                destination = os.path.join(
                    self.processed_dir, os.path.basename(tile_path)
                )
                shutil.move(tile_path, destination)
        logger.info(
            "Scene %s: moved %d tile(s) to processed/.",
            scene.scene_id,
            len(scene.tile_paths),
        )

    # ------------------------------------------------------------------
    # Future extension point: live MOSDAC API
    # ------------------------------------------------------------------
    def fetch_from_api(self, bbox: tuple, start_time: str, end_time: str) -> List[str]:
        """
        Placeholder for a real MOSDAC (or ISRO Bhoonidhi/VEDAS) API client.

        Intended contract once implemented:
            * Authenticate using registered credentials (env vars, not
              hardcoded - see config.py pattern).
            * Query for scenes intersecting ``bbox`` between ``start_time``
              and ``end_time``.
            * Download matching tiles into ``config.MOSDAC_INCOMING_DIR``.
            * Return the list of downloaded file paths.

        Raises
        ------
        NotImplementedError
            Always, until a real API integration is wired in.
        """
        raise NotImplementedError(
            "Live MOSDAC API access is not configured. This system currently "
            "ingests satellite tiles by folder drop into "
            f"'{self.incoming_dir}'. Implement this method with real "
            "credentials to enable automatic acquisition."
        )
