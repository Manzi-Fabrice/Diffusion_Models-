import pandas as pd

# Load CSV
df = pd.read_csv("psnr_degradation_with_scores.csv")

# Sort by degradation rate (lower slope = more resistant)
resistant_images = df.sort_values(by='psnr_slope', ascending=True).head(20)
fragile_images = df.sort_values(by='psnr_slope', ascending=False).head(20)

# Save results
resistant_images.to_csv("top_resistant_images.csv", index=False)
fragile_images.to_csv("top_fragile_images.csv", index=False)

# Print results
print("Most Resistant Images to Noise:")
print(resistant_images[['image_name', 'psnr_slope', 'psnr_delta']])

print("\nMost Fragile Images to Noise:")
print(fragile_images[['image_name', 'psnr_slope', 'psnr_delta']])
