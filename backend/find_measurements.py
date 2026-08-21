import os

SAFE_FOLDER = r"D:\S1\S1D_IW_GRDH_1SDV_20260724T003239_20260724T003252_003808_006DA0_4406_COG.SAFE"

measurement = os.path.join(SAFE_FOLDER, "measurement")

print("Measurement folder:\n")

for file in os.listdir(measurement):
    print(file)