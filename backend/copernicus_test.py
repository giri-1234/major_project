import requests
import os
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv("CDSE_USERNAME")
PASSWORD = os.getenv("CDSE_PASSWORD")

# -----------------------------
# STEP 1 : Get Access Token
# -----------------------------
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

token_response = requests.post(
    TOKEN_URL,
    data={
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": USERNAME,
        "password": PASSWORD,
    },
)

token_response.raise_for_status()

token = token_response.json()["access_token"]

print("✅ Logged in successfully")

# -----------------------------
# STEP 2 : Search Sentinel-1
# -----------------------------

headers = {
    "Authorization": f"Bearer {token}"
}

search_url = "https://sh.dataspace.copernicus.eu/catalog/v1/search"
payload = {
    "collections": ["sentinel-1-grd"],
    "bbox": [68.0, 8.0, 78.0, 18.0],   # Arabian Sea
    "limit": 1,
    "datetime": "2025-07-01T00:00:00Z/2026-12-31T23:59:59Z"
}

response = requests.post(
    search_url,
    headers=headers,
    json=payload
)

print(response.status_code)
print(response.json())

# -----------------------------
# STEP 3 : Get Product UUID
# -----------------------------

product_name = response.json()["features"][0]["id"]

print("\nProduct Name:")
print(product_name)

odata_url = (
    "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    f"?$filter=Name eq '{product_name}'"
)

odata_response = requests.get(odata_url)

odata_response.raise_for_status()

product = odata_response.json()["value"][0]

print("\nUUID:")
print(product["Id"])

import os

DOWNLOAD_URL = (
    f"https://download.dataspace.copernicus.eu/odata/v1/"
    f"Products({product['Id']})/$value"
)

headers = {
    "Authorization": f"Bearer {token}"
}

os.makedirs("downloads", exist_ok=True)

output_file = "downloads/latest_product.zip"

print("\nDownloading Sentinel-1 product...")

with requests.get(DOWNLOAD_URL, headers=headers, stream=True) as r:
    r.raise_for_status()

    with open(output_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

print("\nDownload Completed!")
print(output_file)