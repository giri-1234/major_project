"""
tiling.py
---------
Sliding-window tiling pipeline for large-image oil-spill segmentation.

Splits an image into overlapping tiles, optionally skipping non-water tiles,
and can stitch per-tile masks back into a full-resolution binary mask using
averaged blending in overlap regions.
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.utils import get_logger

logger = get_logger(__name__)


class TilingPipeline:
    """
    Sliding-window tiling for satellite / aerial oil-spill imagery.

    Parameters
    ----------
    tile_size : int
        Width and height of each square tile in pixels.
    stride : int
        Step size between successive tiles. stride < tile_size produces
        overlapping tiles (recommended for clean reconstruction).
    min_water_ratio : float
        Minimum fraction of pixels classified as water for a tile to be
        included in ``extract_tiles``. Tiles with too few water pixels
        (e.g. land areas or sensor noise blocks) are skipped.
    """

    def __init__(
        self,
        tile_size: int = 224,
        stride: int = 112,
        min_water_ratio: float = 0.3,
    ) -> None:
        if tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        if not 0.0 <= min_water_ratio <= 1.0:
            raise ValueError("min_water_ratio must be in [0, 1]")

        self.tile_size = tile_size
        self.stride = stride
        self.min_water_ratio = min_water_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_tiles(self, image: np.ndarray) -> list[dict]:
        """
        Slide a window over ``image`` and return all tiles that pass the
        water-content filter.

        Parameters
        ----------
        image : np.ndarray
            BGR or grayscale image.

        Returns
        -------
        list of dict
            Each dict has keys:
            ``tile``  – the cropped sub-image (np.ndarray),
            ``x``     – left column of the tile in the source image,
            ``y``     – top row of the tile in the source image,
            ``w``     – actual width (may be < tile_size at right edge),
            ``h``     – actual height (may be < tile_size at bottom edge).
        """
        if image is None or image.size == 0:
            raise ValueError("image must not be None or empty")

        h, w = image.shape[:2]
        tiles: list[dict] = []

        y = 0
        while y < h:
            x = 0
            while x < w:
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                tile = image[y:y_end, x:x_end]

                # Pad to full tile_size so every tile has identical shape
                tile = self._pad_tile(tile)

                if self.is_water_tile(tile, self.min_water_ratio):
                    tiles.append(
                        {
                            "tile": tile,
                            "x": x,
                            "y": y,
                            "w": x_end - x,
                            "h": y_end - y,
                        }
                    )

                x += self.stride
            y += self.stride

        logger.debug(
            "extract_tiles: image=%dx%d tile_size=%d stride=%d "
            "total_windows=%d accepted=%d",
            w,
            h,
            self.tile_size,
            self.stride,
            _count_windows(h, w, self.tile_size, self.stride),
            len(tiles),
        )
        return tiles

    def reconstruct_mask(
        self,
        tile_masks: list[dict],
        image_shape: tuple,
    ) -> np.ndarray:
        """
        Stitch per-tile binary masks back into a full-resolution mask.

        Overlapping regions are averaged so that a pixel must appear
        positive in the majority of overlapping tiles to remain positive
        in the final mask.

        Parameters
        ----------
        tile_masks : list of dict
            Each dict must have keys ``mask`` (uint8 binary ndarray),
            ``x``, ``y``, ``w``, ``h`` (integers matching extract_tiles output).
        image_shape : tuple
            ``(height, width)`` of the original image.

        Returns
        -------
        np.ndarray
            Binary uint8 mask of shape ``image_shape``.
        """
        h, w = image_shape[:2]
        accumulator = np.zeros((h, w), dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        for entry in tile_masks:
            mask = entry["mask"]
            x, y = entry["x"], entry["y"]
            tile_w, tile_h = entry["w"], entry["h"]

            # Resize mask to the actual tile footprint (w×h)
            region_mask = cv2.resize(
                mask[:tile_h, :tile_w],
                (tile_w, tile_h),
                interpolation=cv2.INTER_NEAREST,
            )

            y_end = min(y + tile_h, h)
            x_end = min(x + tile_w, w)
            accumulator[y:y_end, x:x_end] += region_mask[:y_end - y, :x_end - x].astype(
                np.float32
            )
            weight[y:y_end, x:x_end] += 1.0

        # Average where tiles overlapped
        valid = weight > 0
        averaged = np.zeros_like(accumulator)
        averaged[valid] = accumulator[valid] / weight[valid]

        # Threshold at 0.5 → majority vote
        binary = (averaged >= 127.5).astype(np.uint8) * 255
        return binary

    def is_water_tile(self, tile: np.ndarray, min_ratio: float) -> bool:
        """
        Heuristic water-content check based on dark pixels in the HSV
        Value channel.

        Dark pixels (HSV-V < 80) are treated as water or oil; a tile is
        considered a "water tile" if the fraction of such pixels meets or
        exceeds ``min_ratio``.

        Parameters
        ----------
        tile : np.ndarray
            A single tile (BGR or grayscale).
        min_ratio : float
            Minimum fraction of dark pixels required.

        Returns
        -------
        bool
        """
        if tile is None or tile.size == 0:
            return False

        if tile.ndim == 2:
            gray = tile
        else:
            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)[:, :, 2]

        dark_pixels = int(np.sum(gray < 80))
        total_pixels = gray.size
        if total_pixels == 0:
            return False

        ratio = dark_pixels / total_pixels
        return ratio >= min_ratio

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad_tile(self, tile: np.ndarray) -> np.ndarray:
        """Zero-pad a tile to ``(tile_size, tile_size)`` if it is smaller."""
        h, w = tile.shape[:2]
        if h == self.tile_size and w == self.tile_size:
            return tile
        if tile.ndim == 3:
            padded = np.zeros(
                (self.tile_size, self.tile_size, tile.shape[2]), dtype=tile.dtype
            )
        else:
            padded = np.zeros((self.tile_size, self.tile_size), dtype=tile.dtype)
        padded[:h, :w] = tile
        return padded


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _count_windows(h: int, w: int, tile_size: int, stride: int) -> int:
    """Count the total number of sliding-window positions."""
    import math

    cols = math.ceil(w / stride) if stride > 0 else 1
    rows = math.ceil(h / stride) if stride > 0 else 1
    return cols * rows
