import pandas as pd

# -------- CONFIG --------
input_csv = "psnr_by_region_summary.csv"
output_csv = "steep_local_drop_images.csv"
drop_threshold = 2.5  # dB drop to consider "steep"

# -------- LOAD DATA --------
df = pd.read_csv(input_csv)

# -------- STEP 1: Identify region columns --------
region_cols = [col for col in df.columns if col != "image_name"]

# -------- STEP 2: Calculate max region PSNR for each image --------
df["max_region_psnr"] = df[region_cols].max(axis=1)

# -------- STEP 3: Calculate drops per region --------
for col in region_cols:
    df[f"{col}_drop"] = df["max_region_psnr"] - df[col]

# -------- STEP 4: Identify if any region has a steep drop --------
drop_cols = [f"{col}_drop" for col in region_cols]
df["has_steep_drop"] = df[drop_cols].max(axis=1) >= drop_threshold

# -------- STEP 5: Filter and Save --------
steep_df = df[df["has_steep_drop"] == True].copy()
steep_df.to_csv(output_csv, index=False)

print(f"Saved {len(steep_df)} images with steep local PSNR drops to: {output_csv}")
