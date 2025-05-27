import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV
df = pd.read_csv("psnr_degradation_with_scores.csv")

# Get top 20 resistant and fragile images
resistant = df.sort_values(by='psnr_slope', ascending=True).head(20)
fragile = df.sort_values(by='psnr_slope', ascending=False).head(20)

# --------- Plot 1: Bar Chart ---------
plt.figure(figsize=(14, 6))
plt.bar(resistant['image_name'], resistant['psnr_slope'], color='green', label='Resistant')
plt.bar(fragile['image_name'], fragile['psnr_slope'], color='red', label='Fragile')
plt.xticks(rotation=90)
plt.xlabel("Image")
plt.ylabel("PSNR Slope")
plt.title("PSNR Degradation Slopes (Top 20 Resistant vs Fragile Images)")
plt.legend()
plt.tight_layout()
plt.savefig("psnr_slope_barplot.png")
plt.close()

# --------- Plot 2: Scatter Plot ---------
plt.figure(figsize=(8, 6))
plt.scatter(df['psnr_slope'], df['psnr_delta'], alpha=0.6)
plt.xlabel("PSNR Slope")
plt.ylabel("Total PSNR Drop (Delta)")
plt.title("PSNR Slope vs PSNR Delta")
plt.grid(True)
plt.tight_layout()
plt.savefig("psnr_slope_vs_delta_scatter.png")
plt.close()

# --------- Plot 3: Line Plot (Degradation Curves) ---------
psnr_cols = [col for col in df.columns if col.startswith('psnr_') and col not in ['psnr_slope', 'psnr_delta']]

plt.figure(figsize=(10, 6))
for _, row in resistant.head(3).iterrows():
    plt.plot(psnr_cols, row[psnr_cols], label=f"Resistant: {row['image_name']}", linestyle='-')
for _, row in fragile.head(3).iterrows():
    plt.plot(psnr_cols, row[psnr_cols], label=f"Fragile: {row['image_name']}", linestyle='--')

plt.xlabel("Noise Step")
plt.ylabel("PSNR")
plt.title("PSNR Degradation Curves")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("psnr_degradation_curves.png")
plt.close()
