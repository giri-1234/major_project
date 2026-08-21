import zipfile
import os

zip_path = "downloads/latest_product.zip"
extract_path = "D:/S1"
os.makedirs(extract_path, exist_ok=True)

print("Extracting...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction Complete!")

print("\nContents:")

for item in os.listdir(extract_path):
    print(item)