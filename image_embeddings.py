import os
import pandas as pd
import torch
import clip
from PIL import Image
import imagehash
from tqdm import tqdm
import numpy as np

# CONFIG
image_dir = "celebahq_256/celeba_hq_256" 
output_clip_path = "clip_embeddings.npy"
output_names_path = "image_names.csv"
output_phash_path = "phash_fingerprints.csv"

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Initialize containers
clip_embeddings = []
image_names = []
phash_entries = []

# Process images
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

for img_name in tqdm(image_files, desc="Processing images"):
    img_path = os.path.join(image_dir, img_name)
    try:
        image = Image.open(img_path).convert("RGB")

        # CLIP embedding
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image_input).cpu().numpy().flatten()
        clip_embeddings.append(embedding)
        image_names.append(img_name)

        # pHash
        phash = str(imagehash.phash(image))
        phash_entries.append((img_name, phash))
    except Exception as e:
        print(f"Error processing {img_name}: {e}")

# Save CLIP embeddings
clip_embeddings = np.array(clip_embeddings)
np.save(output_clip_path, clip_embeddings)
pd.Series(image_names).to_csv(output_names_path, index=False)

# Save pHash fingerprints
phash_df = pd.DataFrame(phash_entries, columns=["image_name", "phash"])
phash_df.to_csv(output_phash_path, index=False)
