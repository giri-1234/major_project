"""
satellite/auth.py
-----------------
CDSE OAuth2 authentication module.

Responsibility: obtain a Bearer access token from the Copernicus Data Space
Ecosystem token endpoint. All other modules receive the token string and do
not need to know how it was obtained.
"""

from __future__ import annotations

import os

import requests

from backend.satellite import AuthenticationError

_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
_CLIENT_ID = "cdse-public"
_TIMEOUT = 30


class Authenticator:
    """Handles CDSE OAuth2 password-grant authentication."""

    def get_token(self) -> str:
        """
        Authenticate with CDSE and return a Bearer access token string.

        Reads ``CDSE_USERNAME`` and ``CDSE_PASSWORD`` from the environment
        (expected to be set via ``.env`` / python-dotenv before this is called).

        Returns
        -------
        str
            The access token value (no 'Bearer' prefix).

        Raises
        ------
        AuthenticationError
            If credentials are missing/empty or if the token endpoint returns
            a non-2xx HTTP response.
        """
        username = os.environ.get("CDSE_USERNAME", "").strip()
        password = os.environ.get("CDSE_PASSWORD", "").strip()

        if not username or not password:
            raise AuthenticationError(
                "CDSE_USERNAME and CDSE_PASSWORD environment variables must both be set "
                "and non-empty before attempting authentication."
            )

        payload = {
            "grant_type": "password",
            "client_id": _CLIENT_ID,
            "username": username,
            "password": password,
        }

        try:
            response = requests.post(_TOKEN_URL, data=payload, timeout=_TIMEOUT)
        except requests.RequestException as exc:
            raise AuthenticationError(
                f"HTTP request to token endpoint failed: {exc}"
            ) from exc

        if not response.ok:
            raise AuthenticationError(
                f"Token endpoint returned HTTP {response.status_code}: {response.text}"
            )

        data = response.json()
        token = data.get("access_token")
        if not token:
            raise AuthenticationError(
                f"Token endpoint response did not contain 'access_token': {data}"
            )

        return token
