
import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv("psnr_summary_combined.csv")

# Pivot the data to have noise_step as columns and image_name as rows
pivot = df.pivot(index="image_name", columns="noise_step", values="psnr")

# Sort columns to ensure correct order
pivot = pivot[[400, 500, 600]]

# Plot PSNR values across noise levels for each image
plt.figure(figsize=(14, 8))
for image in pivot.index:
    plt.plot([400, 500, 600], pivot.loc[image], marker='o', alpha=0.3)

# Labels and title
plt.xlabel("Noise Step", fontsize=14)
plt.ylabel("PSNR", fontsize=14)
plt.title("Robustness per Image: PSNR vs. Noise Level", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save and show
plt.savefig("robustness_per_image_vs_noise.png")
plt.show()
