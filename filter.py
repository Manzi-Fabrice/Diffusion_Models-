import pandas as pd
import os
from PIL import Image
import math

# CONFIG
csv_path = "celebahq_dif_metrics.csv"
image_dir = "celebahq_256/celeba_hq_256"
output_collage = "top_combined_2_percent_collage.jpg"
output_csv = "top_combined_2_percent.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Normalize scores for fair combination
df['lpips_norm'] = (df['lpips'] - df['lpips'].min()) / (df['lpips'].max() - df['lpips'].min())
df['psnr_norm'] = (df['psnr'] - df['psnr'].min()) / (df['psnr'].max() - df['psnr'].min())

# Create a combined score (low LPIPS and high PSNR)
df['combined_score'] = df['lpips_norm'] - df['psnr_norm']  # lower is better

# Sort and take top 2%
df_sorted = df.sort_values(by="combined_score")
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
    print(f"✅ Saved collage with {len(images)} images to '{output_collage}'")
else:
    print("⚠️ No images found to build collage.")
