import pandas as pd
import os
from PIL import Image
import math

# CONFIG
input_csv = "top_2_percent_similarity_analysis.csv"
image_dir = "celebahq_256/celeba_hq_256"  # Directory where images are stored
output_csv = "unique_high_fidelity_candidates.csv"
output_collage = "unique_high_fidelity_collage.jpg"

# Thresholds to define uniqueness
phash_threshold = 10
clip_similarity_threshold = 0.90

# Load the similarity results
df = pd.read_csv(input_csv)

# Filter for unique high-fidelity images
unique_df = df[
    (df["phash_distance"] >= phash_threshold) &
    (df["clip_cosine_similarity"] < clip_similarity_threshold)
]

# Save filtered results
unique_df.to_csv(output_csv, index=False)
print(f"Saved {len(unique_df)} unique high-fidelity samples to '{output_csv}'")

# Create collage of filtered images
image_paths = [
    os.path.join(image_dir, name)
    for name in unique_df["image_name"]
    if os.path.exists(os.path.join(image_dir, name))
]

images = [Image.open(p).convert("RGB") for p in image_paths]

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
    print(f"Collage of {len(images)} images saved to '{output_collage}'")
else:
    print("No images found to build collage.")
