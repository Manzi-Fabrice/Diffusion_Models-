import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the output folder exists
os.makedirs("plots", exist_ok=True)

# Define file paths and region labels
region_files = {
    "top": "celebahq_dif_metrics_top.csv",
    "bottom": "celebahq_dif_metrics_bottom.csv",
    "left": "celebahq_dif_metrics_left.csv",
    "right": "celebahq_dif_metrics_right.csv",
    "center": "celebahq_dif_metrics_center.csv",
    "center_clear": "celebahq_dif_metrics_center_clear.csv",
    "all_noise": "top_combined_2_percent.csv",
}

# Choose which metric to plot: 'psnr', 'lpips', or 'cos_sim'
metric_to_plot = "psnr"

plt.figure(figsize=(10, 6))

# Plot each region
for region, path in region_files.items():
    try:
        df = pd.read_csv(path)
        df_sorted = df.sort_values(by=metric_to_plot).reset_index(drop=True)
        plt.plot(df_sorted.index, df_sorted[metric_to_plot], label=region.capitalize())
    except FileNotFoundError:
        print(f"⚠️ File not found: {path}")
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")

# Finalize the plot
plt.title(f"{metric_to_plot.upper()} Across Different Noise Regions")
plt.xlabel("Sorted Sample Index")
plt.ylabel(metric_to_plot.upper())
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and display
output_path = f"plots/lineplot_{metric_to_plot}_regions.png"
plt.savefig(output_path)
print(f"✅ Plot saved to: {output_path}")
plt.show()
