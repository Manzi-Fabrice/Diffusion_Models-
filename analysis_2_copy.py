import pandas as pd
import matplotlib.pyplot as plt

# Load the full PSNR-by-region dataset
psnr_df = pd.read_csv("psnr_by_region_summary.csv")

# Load the 7 overlapping fragile-memorized images
overlap_df = pd.read_csv("fragile_overlapping_memorized_images.csv")

# Filter for only the overlapping images
filtered_df = psnr_df[psnr_df["image_name"].isin(overlap_df["image_name"])]

# Define the region order for plotting
regions = ["top", "bottom", "left", "right", "center", "center_clear", "all_noise"]

# Plot each image's PSNR across regions
plt.figure(figsize=(12, 6))

for _, row in filtered_df.iterrows():
    psnr_values = [row[region] for region in regions]
    plt.plot(regions, psnr_values, marker='o', label=row["image_name"])

plt.xlabel("Region")
plt.ylabel("PSNR")
plt.title("PSNR by Region for 7 Fragile + Memorized Overlap Images")
plt.legend(title="Image", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True)
plt.show()