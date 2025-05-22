import pandas as pd
import os
from PIL import Image
import math

# CONFIG
csv_path = "celebahq_dif_metrics.csv"  # Your CSV file with LPIPS/PSNR
image_dir = "celebahq_256/celeba_hq_256"  # Folder containing images
output_collage = "worst_1_percent_collage.jpg"
output_csv = "worst_1_percent.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Normalize scores
df['lpips_norm'] = (df['lpips'] - df['lpips'].min()) / (df['lpips'].max() - df['lpips'].min())
df['psnr_norm'] = (df['psnr'] - df['psnr'].min()) / (df['psnr'].max() - df['psnr'].min())

# Combined score: high LPIPS and low PSNR → high combined_score
df['combined_score'] = df['lpips_norm'] + (1 - df['psnr_norm'])  # Higher is worse

# Get top 1% worst reconstructions
df_sorted = df.sort_values(by="combined_score", ascending=False)
top_n = max(1, int(len(df_sorted) * 0.05))
top_df = df_sorted.head(top_n)
top_df.to_csv(output_csv, index=False)

# Load corresponding images
image_paths = [os.path.join(image_dir, name) for name in top_df["image_name"]]
images = [Image.open(p).convert("RGB") for p in image_paths if os.path.exists(p)]

# Create collage
if images:
    img_width, img_height = images[0].size
    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)

    collage = Image.new("RGB", (cols * img_width, rows * img_height), color=(255, 255, 255))

    for idx, img in enumerate(images):
        x = (idx % cols) * img_width
        y = (idx // cols) * img_height
        collage.paste(img, (x, y))

    collage.save(output_collage)
    print(f"Saved collage with {len(images)} worst images to '{output_collage}'")
else:
    print("No images found to build collage.")
