"""
cog_reader.py
-------------
Pure-Python Cloud Optimized GeoTIFF (COG) reader.

Reads ZSTD-compressed COG TIFFs without any native codec DLLs.
Uses only:
  - zstandard  (pure Python ZSTD, no DLLs)
  - numpy      (already required by the project)
  - struct / io (Python stdlib)

This bypasses imagecodecs, zarr, rasterio and GDAL — all of which load
native DLLs blocked by Windows Application Control policy.
"""

from __future__ import annotations

import io
import struct
import numpy as np

try:
    import zstandard as zstd
    _ZSTD_AVAILABLE = True
except ImportError:
    _ZSTD_AVAILABLE = False


# TIFF tag constants
_TAG_IMAGE_WIDTH        = 256
_TAG_IMAGE_LENGTH       = 257
_TAG_BITS_PER_SAMPLE    = 258
_TAG_COMPRESSION        = 259
_TAG_STRIP_OFFSETS      = 273
_TAG_SAMPLES_PER_PIXEL  = 277
_TAG_STRIP_BYTE_COUNTS  = 279
_TAG_TILE_WIDTH         = 322
_TAG_TILE_LENGTH        = 323
_TAG_TILE_OFFSETS       = 324
_TAG_TILE_BYTE_COUNTS   = 325
_TAG_SAMPLE_FORMAT      = 339
_TAG_SUBFILE_TYPE       = 254

_COMPRESSION_ZSTD = 50000
_COMPRESSION_NONE = 1


class CogReadError(Exception):
    pass


def _read_ifd(f, offset, endian):
    """Read one IFD and return a dict of tag -> value."""
    f.seek(offset)
    count = struct.unpack(endian + 'H', f.read(2))[0]
    tags = {}
    for _ in range(count):
        tag, typ, n, raw = struct.unpack(endian + 'HHI4s', f.read(12))
        # Decode value inline or from pointer
        fmt_map = {1:'B', 2:'s', 3:'H', 4:'I', 5:'II', 16:'Q', 8:'h', 9:'i'}
        size_map = {1:1, 2:1, 3:2, 4:4, 5:8, 16:8, 8:2, 9:4}
        fmt = fmt_map.get(typ, 'I')
        sz = size_map.get(typ, 4)
        total = sz * n
        if total <= 4:
            data = raw[:total]
        else:
            ptr = struct.unpack(endian + 'I', raw)[0]
            pos = f.tell()
            f.seek(ptr)
            data = f.read(total)
            f.seek(pos)
        if typ == 2:  # ASCII
            tags[tag] = data.rstrip(b'\x00').decode('ascii', errors='replace')
        elif n == 1 and fmt != 's':
            tags[tag] = struct.unpack(endian + fmt[0], data[:sz])[0]
        else:
            unit = sz
            tags[tag] = [struct.unpack(endian + fmt[0], data[i*unit:(i+1)*unit])[0]
                         for i in range(n)]
    next_ifd = struct.unpack(endian + 'I', f.read(4))[0]
    return tags, next_ifd


def _decompress_zstd(data: bytes, expected_bytes: int) -> bytes:
    if not _ZSTD_AVAILABLE:
        raise CogReadError(
            "zstandard is required. Run: python -m pip install zstandard"
        )
    dctx = zstd.ZstdDecompressor()
    return dctx.decompress(data, max_output_size=expected_bytes * 2)


def read_cog_overview(filepath: str, max_dim: int = 1024, *, normalize: bool = True) -> np.ndarray:
    """
    Read a COG GeoTIFF and return a numpy array.

    Picks the smallest overview level whose longest dimension is >= max_dim/2,
    or the very smallest level if none qualify. This gives a usable image
    without loading the full ~270 MB scene.

    Parameters
    ----------
    filepath : str
        Path to the COG GeoTIFF file.
    max_dim : int
        Target maximum pixel dimension. Default 1024.
    normalize : bool
        When True (default), apply percentile stretch and return uint8.
        When False, return raw backscatter as float32 for downstream
        SAR enhancement stages to handle normalization themselves.

    Returns
    -------
    np.ndarray
        shape (H, W) — uint8 if ``normalize=True``, float32 otherwise.
    """
    with open(filepath, 'rb') as f:
        # Read TIFF header
        byte_order = f.read(2)
        if byte_order == b'II':
            endian = '<'
        elif byte_order == b'MM':
            endian = '>'
        else:
            raise CogReadError(f"Not a TIFF file: {filepath}")

        magic = struct.unpack(endian + 'H', f.read(2))[0]
        if magic != 42:
            raise CogReadError(f"Not a classic TIFF (magic={magic}): {filepath}")

        offset = struct.unpack(endian + 'I', f.read(4))[0]

        # Collect all IFDs (full res + overviews)
        ifds = []
        while offset:
            tags, offset = _read_ifd(f, offset, endian)
            ifds.append(tags)

        if not ifds:
            raise CogReadError("No IFDs found in TIFF.")

        # Pick the best IFD: smallest that is still >= max_dim/2 wide
        chosen = ifds[-1]  # default: smallest
        for ifd in reversed(ifds):
            w = ifd.get(_TAG_IMAGE_WIDTH, 0)
            h = ifd.get(_TAG_IMAGE_LENGTH, 0)
            if max(w, h) >= max_dim // 2:
                chosen = ifd
                break

        w = chosen.get(_TAG_IMAGE_WIDTH, 0)
        h = chosen.get(_TAG_IMAGE_LENGTH, 0)
        bits = chosen.get(_TAG_BITS_PER_SAMPLE, 16)
        compression = chosen.get(_TAG_COMPRESSION, 1)

        if isinstance(bits, list):
            bits = bits[0]

        dtype = np.uint16 if bits == 16 else np.uint8

        # Determine if tiled or stripped
        tile_w = chosen.get(_TAG_TILE_WIDTH)
        tile_h = chosen.get(_TAG_TILE_LENGTH)
        is_tiled = tile_w is not None

        if is_tiled:
            offsets = chosen.get(_TAG_TILE_OFFSETS, [])
            byte_counts = chosen.get(_TAG_TILE_BYTE_COUNTS, [])
            if not isinstance(offsets, list):
                offsets = [offsets]
            if not isinstance(byte_counts, list):
                byte_counts = [byte_counts]
            image = _read_tiled(f, w, h, tile_w, tile_h, offsets,
                                byte_counts, dtype, compression, endian)
        else:
            offsets = chosen.get(_TAG_STRIP_OFFSETS, [])
            byte_counts = chosen.get(_TAG_STRIP_BYTE_COUNTS, [])
            if not isinstance(offsets, list):
                offsets = [offsets]
            if not isinstance(byte_counts, list):
                byte_counts = [byte_counts]
            image = _read_stripped(f, w, h, offsets, byte_counts,
                                   dtype, compression, endian)

    # Return raw float32 or percentile-stretched uint8 depending on caller
    image = image.astype(np.float32)
    if not normalize:
        return image

    valid = image[image > 0]
    if valid.size > 0:
        p2 = float(np.percentile(valid, 2))
        p98 = float(np.percentile(valid, 98))
    else:
        p2, p98 = 0.0, 1.0
    image = np.clip(image, p2, p98)
    image = ((image - p2) / (p98 - p2 + 1e-10) * 255).astype(np.uint8)
    return image


def _decompress_tile(data: bytes, n_bytes: int, compression: int,
                     endian: str) -> bytes:
    if compression == _COMPRESSION_NONE:
        return data
    elif compression == _COMPRESSION_ZSTD:
        return _decompress_zstd(data, n_bytes)
    else:
        raise CogReadError(
            f"Unsupported compression: {compression}. "
            "Only uncompressed (1) and ZSTD (50000) are supported."
        )


def _read_tiled(f, width, height, tile_w, tile_h, offsets,
                byte_counts, dtype, compression, endian) -> np.ndarray:
    tiles_x = (width + tile_w - 1) // tile_w
    tiles_y = (height + tile_h - 1) // tile_h
    image = np.zeros((height, width), dtype=dtype)
    tile_bytes = tile_w * tile_h * dtype(0).itemsize

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            idx = ty * tiles_x + tx
            if idx >= len(offsets):
                break
            f.seek(offsets[idx])
            raw = f.read(byte_counts[idx])
            raw = _decompress_tile(raw, tile_bytes, compression, endian)
            tile = np.frombuffer(raw[:tile_bytes], dtype=dtype)
            # Fix byte order if needed
            if (endian == '>' and dtype == np.uint16):
                tile = tile.byteswap()
            tile = tile.reshape(tile_h, tile_w)
            y0 = ty * tile_h
            x0 = tx * tile_w
            y1 = min(y0 + tile_h, height)
            x1 = min(x0 + tile_w, width)
            image[y0:y1, x0:x1] = tile[:y1-y0, :x1-x0]

    return image


def _read_stripped(f, width, height, offsets, byte_counts,
                   dtype, compression, endian) -> np.ndarray:
    rows_per_strip = max(1, height // len(offsets)) if offsets else height
    image = np.zeros((height, width), dtype=dtype)
    row = 0
    for i, (off, bc) in enumerate(zip(offsets, byte_counts)):
        f.seek(off)
        raw = f.read(bc)
        n_bytes = min(rows_per_strip, height - row) * width * dtype(0).itemsize
        raw = _decompress_tile(raw, n_bytes, compression, endian)
        n_rows = len(raw) // (width * dtype(0).itemsize)
        strip = np.frombuffer(raw[:n_rows * width * dtype(0).itemsize], dtype=dtype)
        if endian == '>' and dtype == np.uint16:
            strip = strip.byteswap()
        strip = strip.reshape(n_rows, width)
        image[row:row + n_rows] = strip
        row += n_rows
    return image
