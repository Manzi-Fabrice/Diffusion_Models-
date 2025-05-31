import pandas as pd
import matplotlib.pyplot as plt

# Load PSNR summary and 7-image list
df = pd.read_csv("psnr_by_region_summary.csv")
targets = pd.read_csv("fragile_overlapping_memorized_images.csv")

# Filter only the 7 images
df_filtered = df[df["image_name"].isin(targets["image_name"])].copy()
regions = ["top", "bottom", "left", "right", "center", "center_clear", "all_noise"]
df_filtered["most_fragile_region"] = df_filtered[regions].idxmin(axis=1)

# Group images by region
region_dict = df_filtered.groupby("most_fragile_region")["image_name"].apply(list).to_dict()

# Setup canvas layout for each region
layout_positions = {
    "top": (0.5, 0.85),
    "bottom": (0.5, 0.15),
    "left": (0.15, 0.5),
    "right": (0.85, 0.5),
    "center": (0.5, 0.5),
    "center_clear": (0.5, 0.35),
    "all_noise": (0.85, 0.85)
}

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
plt.title("Image Fragility Clustered by Lowest-PSNR Region", fontsize=14)

# Draw region containers and place image names
for region, (x, y) in layout_positions.items():
    ax.text(x, y + 0.05, region.upper(), fontsize=10, ha="center", weight="bold", bbox=dict(facecolor='lightgray', alpha=0.6))
    names = region_dict.get(region, [])
    for i, name in enumerate(names):
        ax.text(x, y - i * 0.04, name, fontsize=9, ha="center", color="purple")

# Save
plt.tight_layout()
plt.savefig("region_cluster_layout.png")

