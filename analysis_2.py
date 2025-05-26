import pandas as pd
import os

# Mapping of region to CSV file
region_files = {
    "top": "celebahq_dif_metrics_top.csv",
    "bottom": "celebahq_dif_metrics_bottom.csv",
    "left": "celebahq_dif_metrics_left.csv",
    "right": "celebahq_dif_metrics_right.csv",
    "center": "celebahq_dif_metrics_center.csv",
    "center_clear": "celebahq_dif_metrics_center_clear.csv",
    "all_noise": "top_combined_2_percent.csv",  # Assumes this also has PSNR
}

# Dictionary to collect data
combined_data = {}

for region, filepath in region_files.items():
    try:
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            image_name = os.path.basename(row["image_name"])  # Strip path
            psnr = row["psnr"]

            if image_name not in combined_data:
                combined_data[image_name] = {}
            combined_data[image_name][region] = psnr
    except FileNotFoundError:
        print(f"File not found: {filepath}")

# Convert to DataFrame
summary_df = pd.DataFrame.from_dict(combined_data, orient="index")
summary_df.index.name = "image_name"
summary_df.reset_index(inplace=True)

# Save to CSV
summary_df.to_csv("psnr_by_region_summary.csv", index=False)
print("✅ Saved to psnr_by_region_summary.csv")
