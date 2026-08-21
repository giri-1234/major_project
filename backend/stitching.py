"""
stitching.py
------------
Merges multiple overlapping satellite image tiles into a single stitched
scene before it enters the preprocessing/prediction pipeline.

Two strategies are used, in order of preference:

1. ``cv2.Stitcher`` (OpenCV's built-in feature-matching + blending
   pipeline) - robust for photographic overlap, works out of the box.
2. A manual ORB + homography fallback for the cases ``cv2.Stitcher``
   fails on (e.g. very low-texture tiles, only 2 images, etc).

Supported formats: PNG, JPEG, GeoTIFF (``.tif``/``.tiff``).
GeoTIFF tiles are read through ``rasterio`` when it is installed (so real
geo-referenced mosaicking can reproject tiles correctly); if ``rasterio``
is not available, GeoTIFFs are read as plain raster arrays via OpenCV,
which is sufficient for pixel-level stitching but ignores geo-transform.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from backend import config
from backend.utils import get_logger

logger = get_logger(__name__)

# Raise Pillow's decompression-bomb limit for large satellite GeoTIFFs.
# Sentinel-1 scenes are ~232 million pixels — well above the default 178M limit.
Image.MAX_IMAGE_PIXELS = None  # disable limit for trusted satellite data

try:
    import rasterio  # type: ignore
    _RASTERIO_AVAILABLE = True
except ImportError:
    _RASTERIO_AVAILABLE = False

# Maximum pixel dimension when reading a large SAR GeoTIFF.
# The image is downscaled to fit within this box while preserving aspect ratio.
# 4096 px gives enough detail for oil-slick detection while staying memory-safe.
_SAR_MAX_DIM = 4096


class StitchingError(Exception):
    """Raised when tiles cannot be stitched into a single scene."""


@dataclass
class StitchResult:
    stitched_image: np.ndarray
    tile_count: int
    method_used: str
    output_path: str


def _read_tile(filepath: str) -> np.ndarray:
    """
    Load a single tile into a BGR numpy array.

    For large SAR GeoTIFFs (Sentinel-1 scenes) that exceed safe memory limits,
    the image is downscaled to fit within _SAR_MAX_DIM pixels on the longest
    side. This preserves enough spatial detail for oil-slick detection while
    avoiding memory exhaustion on desktop hardware.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".tif", ".tiff"):
        return _read_sar_tiff(filepath)

    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if image is None:
        raise StitchingError(f"Could not read tile: {filepath}")
    return image


def _read_sar_tiff(filepath: str) -> np.ndarray:
    """
    Read a COG GeoTIFF (Sentinel-1 product) using tifffile + imagecodecs.
    Picks a reduced resolution overview level that fits within _SAR_MAX_DIM
    to keep memory usage safe while preserving enough spatial detail.
    Converts to 8-bit BGR for the rest of the pipeline.
    """
    try:
        import tifffile  # noqa: WPS433
    except ImportError as exc:
        raise StitchingError(
            "tifffile is required to read COG GeoTIFFs. "
            "Run: python -m pip install tifffile imagecodecs"
        ) from exc

    # Pick the best overview level
    with tifffile.TiffFile(filepath) as tif:
        levels = tif.series[0].levels
        n_levels = len(levels)

    # Find the smallest level that is still >= _SAR_MAX_DIM / 2 in width
    # so we get a reasonable amount of detail
    chosen_level = n_levels - 1  # default: smallest
    with tifffile.TiffFile(filepath) as tif:
        for i, lv in enumerate(tif.series[0].levels):
            shape = lv.shape  # (H, W) or (bands, H, W)
            w = shape[-1]
            h = shape[-2]
            if max(h, w) <= _SAR_MAX_DIM:
                chosen_level = i
                break

    logger.info("Reading COG level %d of %d from '%s'",
                chosen_level, n_levels, os.path.basename(filepath))

    try:
        image = tifffile.imread(filepath, series=0, level=chosen_level)
    except Exception as exc:
        raise StitchingError(
            f"Could not read SAR GeoTIFF '{filepath}': {exc}"
        ) from exc

    logger.info("COG loaded: shape=%s dtype=%s", image.shape, image.dtype)

    # tifffile may return (H, W) or (C, H, W) — normalize to (H, W, C)
    if image.ndim == 3 and image.shape[0] <= 4:
        image = np.moveaxis(image, 0, -1)

    # Downscale if still too large
    h, w = image.shape[0], image.shape[1]
    if max(h, w) > _SAR_MAX_DIM:
        scale = _SAR_MAX_DIM / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
        logger.info("Further downscaled to %dx%d", int(w * scale), int(h * scale))

    # Convert to uint8 with percentile stretch
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        valid = image[image > 0]
        if valid.size > 0:
            p2 = float(np.percentile(valid, 2))
            p98 = float(np.percentile(valid, 98))
        else:
            p2, p98 = 0.0, 1.0
        image = np.clip(image, p2, p98)
        image = ((image - p2) / (p98 - p2 + 1e-10) * 255).astype(np.uint8)

    # Ensure 3-channel BGR
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]

    return image


def load_tiles(filepaths: List[str]) -> List[np.ndarray]:
    """Load a list of tile file paths into BGR numpy arrays."""
    if not filepaths:
        raise StitchingError("No tile file paths were provided.")
    return [_read_tile(path) for path in filepaths]


def _stitch_with_opencv(tiles: List[np.ndarray]) -> Optional[np.ndarray]:
    """Try OpenCV's built-in panorama stitcher first."""
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    except AttributeError:  # older OpenCV API
        stitcher = cv2.createStitcher()  # type: ignore[attr-defined]

    status, stitched = stitcher.stitch(tiles)
    if status == cv2.Stitcher_OK:
        return stitched

    logger.warning("cv2.Stitcher failed with status code %s; falling back.", status)
    return None


def _stitch_pair_manual(base: np.ndarray, next_tile: np.ndarray) -> np.ndarray:
    """
    Stitch two images using ORB feature detection + homography warping.
    Used as a fallback when ``cv2.Stitcher`` cannot find enough overlap
    (common with low-texture ocean/water tiles).
    """
    cfg = config.STITCHING
    orb = cv2.ORB_create(nfeatures=cfg.orb_features)

    gray_base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    gray_next = cv2.cvtColor(next_tile, cv2.COLOR_BGR2GRAY)

    kp_base, des_base = orb.detectAndCompute(gray_base, None)
    kp_next, des_next = orb.detectAndCompute(gray_next, None)

    if des_base is None or des_next is None:
        raise StitchingError("Not enough features detected for manual stitching.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(des_next, des_base, k=2)

    good_matches = [
        m for m, n in raw_matches if m.distance < cfg.lowe_ratio * n.distance
    ]

    if len(good_matches) < cfg.min_match_count:
        raise StitchingError(
            f"Only {len(good_matches)} good matches found "
            f"(need >= {cfg.min_match_count}); tiles do not overlap enough."
        )

    src_pts = np.float32([kp_next[m.queryIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )
    dst_pts = np.float32([kp_base[m.trainIdx].pt for m in good_matches]).reshape(
        -1, 1, 2
    )

    homography, _ = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, cfg.ransac_reproj_threshold
    )
    if homography is None:
        raise StitchingError("Homography estimation failed between tiles.")

    h_base, w_base = base.shape[:2]
    h_next, w_next = next_tile.shape[:2]
    canvas_width = w_base + w_next
    canvas_height = max(h_base, h_next)

    warped = cv2.warpPerspective(next_tile, homography, (canvas_width, canvas_height))
    warped[0:h_base, 0:w_base] = base
    return warped


def _stitch_manual_sequence(tiles: List[np.ndarray]) -> np.ndarray:
    """Fold a list of tiles into one image by repeatedly stitching pairs."""
    result = tiles[0]
    for tile in tiles[1:]:
        result = _stitch_pair_manual(result, tile)
    return result


def stitch_tiles(
    filepaths: List[str], output_path: Optional[str] = None
) -> StitchResult:
    """
    Stitch a set of tile images into one scene.

    Parameters
    ----------
    filepaths : list[str]
        Paths to the tile images (2 or more). Order does not need to be
        precise - OpenCV's stitcher determines overlap automatically; the
        manual fallback stitches left-to-right in the given order.
    output_path : str, optional
        Where to save the stitched result. Defaults to a generated path
        inside ``config.UPLOAD_FOLDER``.

    Returns
    -------
    StitchResult
    """
    tiles = load_tiles(filepaths)

    if len(tiles) == 1:
        logger.info("Only one tile provided; skipping stitching.")
        stitched, method = tiles[0], "single-tile-passthrough"
    else:
        stitched = _stitch_with_opencv(tiles)
        method = "opencv_stitcher"
        if stitched is None:
            stitched = _stitch_manual_sequence(tiles)
            method = "manual_orb_homography"

    if output_path is None:
        output_path = os.path.join(config.UPLOAD_FOLDER, "stitched_scene.png")

    cv2.imwrite(output_path, stitched)
    logger.info(
        "Stitched %d tile(s) using '%s' -> %s", len(tiles), method, output_path
    )

    return StitchResult(
        stitched_image=stitched,
        tile_count=len(tiles),
        method_used=method,
        output_path=output_path,
    )
