import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("psnr_degradation_with_scores.csv")

plt.figure(figsize=(12, 8))

# Scatter plot with color based on slope
sc = plt.scatter(df["psnr_delta"], df["psnr_auc"], 
                 c=df["psnr_slope"], cmap="coolwarm", s=40)

# Add annotations (image names)
for _, row in df.iterrows():
    plt.annotate(
        row["image_name"].split(".")[0],  # show just the numeric ID
        (row["psnr_delta"], row["psnr_auc"]),
        fontsize=7,
        alpha=0.7
    )

# Labels and styling
plt.xlabel("PSNR Total Drop (Delta)")
plt.ylabel("PSNR AUC")
plt.title("Image Sensitivity to Noise")
plt.colorbar(sc, label="PSNR Slope (Rate of Decay)")
plt.grid(True)
plt.tight_layout()
plt.savefig("psnr_sensitivity_labeled.png")
plt.show()
