
import pandas as pd
import matplotlib.pyplot as plt

# Load PSNR data
csv_file = "psnr_by_region_summary.csv"  # Make sure this file is in the same directory
df = pd.read_csv(csv_file)

# Compute average PSNR per noise region
average_psnr = df.drop(columns=["image_name"]).mean().sort_values()

# Plot simple line graph
plt.figure(figsize=(10, 6))
plt.plot(average_psnr.index, average_psnr.values, marker='o', linestyle='-', color='blue')
plt.title("Average PSNR by Noise Region")
plt.xlabel("Noise Region")
plt.ylabel("Average PSNR")
plt.grid(True)
plt.tight_layout()
plt.show()
