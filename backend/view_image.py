import rasterio
import matplotlib.pyplot as plt

image_path = r"D:\S1\S1D_IW_GRDH_1SDV_20260724T003239_20260724T003252_003808_006DA0_4406_COG.SAFE\measurement\s1d-iw-grd-vv-20260724t003239-20260724t003252-003808-006da0-001-cog.tiff"

with rasterio.open(image_path) as src:
    image = src.read(1)

print("Image Shape:", image.shape)
print("Data Type:", image.dtype)

plt.figure(figsize=(10,10))
plt.imshow(image, cmap="gray")
plt.title("Sentinel-1 VV Band")
plt.colorbar()
plt.show()