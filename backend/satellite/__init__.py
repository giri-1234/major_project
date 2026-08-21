"""
satellite/__init__.py
---------------------
Exception taxonomy for the satellite acquisition package.

All four classes are exported from here so callers import from a single
location, e.g.::

    from backend.satellite import SatelliteAcquisitionError, AuthenticationError
"""


class SatelliteAcquisitionError(Exception):
    """Base exception for all satellite acquisition failures."""


class AuthenticationError(SatelliteAcquisitionError):
    """Raised when CDSE OAuth2 authentication fails or credentials are missing."""


class ProductNotFoundError(SatelliteAcquisitionError):
    """Raised when no matching Sentinel-1 product is found in CDSE catalog."""


class DownloadError(SatelliteAcquisitionError):
    """Raised when product download or extraction fails."""


__all__ = [
    "SatelliteAcquisitionError",
    "AuthenticationError",
    "ProductNotFoundError",
    "DownloadError",
]
