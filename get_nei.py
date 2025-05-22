import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# CONFIG
csv_path = "worst_similarity_analysis.csv" 
image_dir = "celebahq_256/celeba_hq_256"
output_collage_path = "low_triplet_collage.jpg"

# Load CSV
df = pd.read_csv(csv_path)

# Load and group images for triplet collage
triplets = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing triplets"):
    names = [row["image_name"], row["closest_phash_image"], row["closest_clip_image"]]
    images = []
    for name in names:
        path = os.path.join(image_dir, name)
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            images.append(img)
        else:
            print(f"⚠️ Missing: {name}")
            images.append(Image.new("RGB", (256, 256), (255, 255, 255)))  # blank placeholder
    triplets.append(images)

# Create collage: each row is [original, pHash match, CLIP match]
if triplets:
    img_width, img_height = triplets[0][0].size
    collage_width = img_width * 3
    collage_height = img_height * len(triplets)
    collage = Image.new("RGB", (collage_width, collage_height), color=(255, 255, 255))

    for idx, triplet in enumerate(triplets):
        for col, img in enumerate(triplet):
            x = col * img_width
            y = idx * img_height
            collage.paste(img, (x, y))

    collage.save(output_collage_path)
    print(f"✅ Collage saved to {output_collage_path}")
else:
    print("No valid triplets to create collage.")
