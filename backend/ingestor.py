"""
ingestor.py
-----------
Scene ingestion for the Sentinel-1 / Copernicus satellite pipeline.

This module watches the local incoming folder where the satellite acquisition
system (``SatelliteManager``) drops processed GeoTIFF files after download
and SNAP preprocessing. It groups available images into a ``Scene`` and
hands them to the stitching/prediction pipeline.

The folder-based strategy means any image dropped into
``datasets/sentinel1/incoming/`` is automatically picked up when the
operator clicks "Scan Incoming" — whether it came from the 🛰 button or
was copied manually for testing.
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
class Scene:
    """One acquisition grouping — one or more image files ready to process."""

    scene_id: str
    tile_paths: List[str]
    acquired_at: str


class SceneIngestor:
    """
    Watches the Sentinel-1 incoming folder and surfaces new scenes to the
    processing pipeline.

    Images are placed here by ``SatelliteManager.download_latest()`` after
    SNAP preprocessing produces a valid GeoTIFF. They can also be placed
    manually for testing.
    """

    def __init__(
        self,
        incoming_dir: str = config.SENTINEL1_INCOMING_DIR,
        processed_dir: str = config.SENTINEL1_PROCESSED_DIR,
    ):
        self.incoming_dir = incoming_dir
        self.processed_dir = processed_dir
        os.makedirs(self.incoming_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def scan_incoming_folder(self) -> List[str]:
        """Return full paths of all valid images waiting in the incoming folder."""
        candidates = sorted(os.listdir(self.incoming_dir))
        image_paths = [
            os.path.join(self.incoming_dir, name)
            for name in candidates
            if is_allowed_image(name)
            and os.path.isfile(os.path.join(self.incoming_dir, name))
        ]
        logger.info("Incoming folder scan found %d image(s).", len(image_paths))
        return image_paths

    def fetch_new_images(self) -> Scene:
        """
        Build a Scene from everything currently in the incoming folder.

        Raises
        ------
        FileNotFoundError
            If no eligible images are found.
        """
        tile_paths = self.scan_incoming_folder()
        if not tile_paths:
            raise FileNotFoundError(
                f"No images found in '{self.incoming_dir}'. "
                f"Use the 🛰 button to download a Sentinel-1 image, or drop "
                f"a PNG/JPEG/TIFF file there manually to test."
            )

        scene_id = datetime.now().strftime("S1_%Y%m%d_%H%M%S")
        return Scene(
            scene_id=scene_id,
            tile_paths=tile_paths,
            acquired_at=datetime.utcnow().isoformat() + "Z",
        )

    def mark_processed(self, scene: Scene) -> None:
        """Move a scene's tiles to processed/ once the pipeline completes."""
        for tile_path in scene.tile_paths:
            if os.path.isfile(tile_path):
                destination = os.path.join(
                    self.processed_dir, os.path.basename(tile_path)
                )
                shutil.move(tile_path, destination)
        logger.info(
            "Scene %s: moved %d file(s) to processed/.",
            scene.scene_id,
            len(scene.tile_paths),
        )
