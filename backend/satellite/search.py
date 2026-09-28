"""
satellite/search.py
-------------------
CDSE product search module.

Search backend: Copernicus OData catalog
  https://catalogue.dataspace.copernicus.eu/odata/v1/Products

The previous implementation used the Sentinel Hub STAC endpoint
(sh.dataspace.copernicus.eu/catalog/v1/search) which requires a paid
Sentinel Hub processing-unit subscription and consistently returns empty
feature arrays on free CDSE accounts, even when products exist.

The OData catalog is the official free-tier Copernicus product index.
It requires no subscription, works with or without authentication, and
is the same backend used by the Copernicus Browser. Authentication is
still accepted (and passed when available) but is not required for search.

Public interface (unchanged from STAC version):
  search_bbox(bbox, token, max_days, limit) -> dict
    { products, days_searched, expanded, suggestions }
  find_latest_product(token, bbox)           -> str  (product name)
  resolve_product_uuid(product_name, token)  -> dict { id, name }
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests

from backend.satellite import ProductNotFoundError

# ---------------------------------------------------------------------------
# Endpoint constants
# ---------------------------------------------------------------------------
_ODATA_URL  = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# Kept for reference / future re-enablement; not used for search any more.
_STAC_URL   = "https://sh.dataspace.copernicus.eu/catalog/v1/search"

# Arabian Sea — default bbox used by SatelliteManager when no user bbox given
_ARABIAN_SEA_BBOX = [68.0, 8.0, 78.0, 18.0]

# Expanding search windows (days). Tried in order; stops at first non-empty result.
_SEARCH_WINDOW_STEPS = [30, 60, 90, 180]

_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Main searcher class
# ---------------------------------------------------------------------------

class ProductSearcher:
    """
    Queries the Copernicus OData catalog to find Sentinel-1 GRD products.
    """

    # ------------------------------------------------------------------
    # Public: adaptive bbox search (called by the map UI route)
    # ------------------------------------------------------------------

    def search_bbox(
        self,
        bbox: List[float],
        token: str,
        max_days: int = 180,
        limit: int = 20,
    ) -> dict:
        """
        Search the OData catalog for Sentinel-1 GRD products intersecting
        ``bbox`` with automatic search-window expansion.

        Tries increasing windows (30 → 60 → 90 → 180 days), stopping as
        soon as at least one product is found.

        Parameters
        ----------
        bbox : [lon_min, lat_min, lon_max, lat_max]
        token : str
            CDSE Bearer token (optional for OData; passed for consistency).
        max_days : int
            Hard ceiling on the search window.
        limit : int
            Maximum products to return.

        Returns
        -------
        dict
            products       – list of normalised product dicts
            days_searched  – window (days) that returned products, or max tried
            expanded       – True if window was widened beyond 30 days
            suggestions    – non-empty list only when products is empty
        """
        # Cap windows at max_days
        windows = [w for w in _SEARCH_WINDOW_STEPS if w <= max_days]
        if not windows:
            windows = _SEARCH_WINDOW_STEPS[:]

        # OData accepts auth but works without it too
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        now = datetime.now(tz=timezone.utc)
        products: List[dict] = []
        days_searched = windows[0]
        expanded = False

        for days in windows:
            days_searched = days
            if days > _SEARCH_WINDOW_STEPS[0]:
                expanded = True

            start_dt = now - timedelta(days=days)
            odata_filter = self._build_odata_filter(bbox, start_dt, now)

            params = {
                "$filter":  odata_filter,
                "$top":     limit,
                "$orderby": "ContentDate/Start desc",
            }

            try:
                resp = requests.get(
                    _ODATA_URL,
                    params=params,
                    headers=headers,
                    timeout=_TIMEOUT,
                )
            except requests.RequestException as exc:
                return {
                    "products":     [],
                    "days_searched": days_searched,
                    "expanded":     expanded,
                    "suggestions":  [
                        f"Network error reaching Copernicus catalog: {exc}",
                        "Verify your internet connection and try again.",
                    ],
                }

            if not resp.ok:
                return {
                    "products":     [],
                    "days_searched": days_searched,
                    "expanded":     expanded,
                    "suggestions":  [
                        f"OData catalog returned HTTP {resp.status_code}. "
                        f"Response: {resp.text[:300]}",
                    ],
                }

            value = resp.json().get("value", [])
            if value:
                products = [self._odata_entry_to_product(e) for e in value]
                break
            # No GRD products yet — widen the window and retry

        if not products:
            return {
                "products":     [],
                "days_searched": days_searched,
                "expanded":     expanded,
                "suggestions":  self._build_suggestions(bbox, days_searched),
            }

        return {
            "products":     products,
            "days_searched": days_searched,
            "expanded":     expanded,
            "suggestions":  [],
        }

    # ------------------------------------------------------------------
    # Public: legacy method used by SatelliteManager
    # ------------------------------------------------------------------

    def find_latest_product(
        self,
        token: str,
        bbox: Optional[List[float]] = None,
    ) -> str:
        """
        Return the name of the most recent Sentinel-1 GRD product for the
        given bbox (defaults to Arabian Sea). Raises ProductNotFoundError
        if nothing is found within 30 days.
        """
        search_bbox = bbox or _ARABIAN_SEA_BBOX
        result = self.search_bbox(search_bbox, token, max_days=30, limit=1)
        if not result["products"]:
            raise ProductNotFoundError(
                f"No Sentinel-1 GRD products found for bbox={search_bbox} "
                "in the past 30 days."
            )
        return result["products"][0]["id"]

    # ------------------------------------------------------------------
    # Public: OData UUID resolution (unchanged from previous version)
    # ------------------------------------------------------------------

    def resolve_product_uuid(self, product_name: str, token: str) -> dict:
        """
        Resolve a Sentinel-1 product name to its CDSE OData UUID.

        Returns dict { "id": <uuid>, "name": <product_name> }
        Raises ProductNotFoundError if no entry is found.
        """
        params = {
            "$filter": f"Name eq '{product_name}'",
            "$top":    1,
        }
        try:
            resp = requests.get(_ODATA_URL, params=params, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            raise requests.RequestException(
                f"OData products request failed: {exc}"
            ) from exc

        resp.raise_for_status()
        value = resp.json().get("value", [])

        if not value:
            raise ProductNotFoundError(
                f"No OData entry found for product name '{product_name}'. "
                "The product may not yet be indexed or the name may be incorrect."
            )

        entry = value[0]
        return {"id": entry["Id"], "name": entry["Name"]}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_odata_filter(
        bbox: List[float],
        start_dt: datetime,
        end_dt: datetime,
    ) -> str:
        """
        Build an OData $filter string for Sentinel-1 GRD products (both IW and
        EW modes) intersecting a bounding box within a date range.

        Automatically applies a buffer if the ROI is smaller than 0.4 degrees
        (~45 km) to ensure adjacent orbital swaths are not missed.
        """
        lon_min, lat_min, lon_max, lat_max = bbox

        # Auto-buffer small bounding boxes to catch adjacent satellite swaths
        if abs(lon_max - lon_min) < 0.4 or abs(lat_max - lat_min) < 0.4:
            buf = 0.35
            lon_min -= buf
            lat_min -= buf
            lon_max += buf
            lat_max += buf

        # Closed WKT ring: SW → SE → NE → NW → SW
        wkt = (
            f"POLYGON(("
            f"{lon_min:.4f} {lat_min:.4f},"
            f"{lon_max:.4f} {lat_min:.4f},"
            f"{lon_max:.4f} {lat_max:.4f},"
            f"{lon_min:.4f} {lat_max:.4f},"
            f"{lon_min:.4f} {lat_min:.4f}"
            f"))"
        )

        start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        end_str   = end_dt.strftime(  "%Y-%m-%dT%H:%M:%S.000Z")

        # Matches both IW and EW GRD products (Interferometric Wide & Extra-Wide)
        return (
            "Collection/Name eq 'SENTINEL-1' "
            "and (contains(Name,'GRDH') or contains(Name,'GRDM') or contains(Name,'_EW_') or contains(Name,'_IW_')) "
            f"and ContentDate/Start gt {start_str} "
            f"and ContentDate/Start lt {end_str} "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"
        )

    @staticmethod
    def _odata_entry_to_product(entry: dict) -> dict:
        """
        Normalise a CDSE OData product entry into the flat dict the UI expects,
        including orbital swath footprint geometry if present.
        """
        name = entry.get("Name", "")

        # Date from ContentDate.Start e.g. "2026-08-05T00:32:39.000Z"
        raw_date = (entry.get("ContentDate") or {}).get("Start", "")
        date_str = raw_date[:10] if raw_date else ""

        # Orbit direction from attributes list
        attrs = entry.get("Attributes", []) or []
        orbit = ""
        mode  = "IW"
        for attr in attrs:
            attr_name  = attr.get("Name", "")
            attr_value = attr.get("Value", "")
            if attr_name == "orbitDirection":
                orbit = str(attr_value).upper()
            if attr_name == "operationalMode":
                mode = str(attr_value).upper()

        # File size in MB from ContentLength (bytes)
        size_bytes = entry.get("ContentLength", 0) or 0
        size_mb    = f"~{round(size_bytes / 1_048_576)}" if size_bytes else "~474"

        # Extract satellite footprint polygon (GeoJSON geometry dict)
        footprint = entry.get("GeoFootprint")

        return {
            "id":        name,          # product name used as ID for download
            "name":      name,
            "date":      date_str,
            "orbit":     orbit or "—",
            "platform":  "SENTINEL-1",
            "mode":      mode,
            "size_mb":   size_mb,
            "footprint": footprint,
        }

    @staticmethod
    def _build_suggestions(bbox: List[float], days_searched: int) -> List[str]:
        """Return contextual suggestions when no products are found."""
        lon_min, lat_min, lon_max, lat_max = bbox
        width_deg  = abs(lon_max - lon_min)
        height_deg = abs(lat_max - lat_min)
        area_km2   = width_deg * height_deg * 111 * 111

        suggestions = []

        if area_km2 < 5_000:
            suggestions.append(
                f"Your selected area is very small (~{area_km2:.0f} km²). "
                "Sentinel-1 IW scenes cover ~250×170 km — draw a larger rectangle."
            )
        else:
            suggestions.append(
                f"No Sentinel-1 pass intersected this area in the past "
                f"{days_searched} days. Try expanding your rectangle."
            )

        if lat_min > 60 or lat_max < -60:
            suggestions.append(
                "Polar regions have irregular SAR coverage. "
                "Try a mid-latitude ocean area (–60° to +60°)."
            )

        suggestions.append(
            "Select a nearby open-ocean area — coast-hugging boxes often "
            "miss passes because scene footprints favour offshore swaths."
        )
        suggestions.append(
            "Use one of the preset regions (Arabian Sea, Bay of Bengal) "
            "which have frequent Sentinel-1 coverage."
        )

        return suggestions
