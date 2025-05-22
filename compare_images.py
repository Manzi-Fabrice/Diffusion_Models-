import pandas as pd
import numpy as np
import imagehash
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Load top image names
top_df = pd.read_csv("worst_1_percent.csv")
top_image_names = top_df["image_name"].tolist()

# Load pHash fingerprints
phash_df = pd.read_csv("phash_fingerprints.csv")
phash_dict = dict(zip(phash_df["image_name"], phash_df["phash"]))

# Load CLIP embeddings
clip_embeddings = np.load("clip_embeddings.npy")
clip_image_names = pd.read_csv("image_names.csv")["0"].tolist()
clip_dict = dict(zip(clip_image_names, clip_embeddings))

# Function to compute hamming distance between pHash strings
def phash_distance(hash1, hash2):
    return imagehash.hex_to_hash(hash1) - imagehash.hex_to_hash(hash2)

# For each top image, find the most similar training image using pHash and CLIP
results = []

for name in tqdm(top_image_names, desc="Comparing"):
    if name not in phash_dict or name not in clip_dict:
        continue

    phash_ref = phash_dict[name]
    clip_ref = clip_dict[name].reshape(1, -1)

    # pHash comparison
    phash_dists = [(img, phash_distance(phash_ref, ph)) for img, ph in phash_dict.items() if img != name]
    phash_closest = sorted(phash_dists, key=lambda x: x[1])[0]  # closest by pHash

    # CLIP similarity
    clip_matrix = np.array([v for k, v in clip_dict.items() if k != name])
    clip_keys = [k for k in clip_dict if k != name]
    sims = cosine_similarity(clip_ref, clip_matrix)[0]
    max_idx = np.argmax(sims)
    clip_closest = (clip_keys[max_idx], sims[max_idx])

    results.append({
        "image_name": name,
        "closest_phash_image": phash_closest[0],
        "phash_distance": phash_closest[1],
        "closest_clip_image": clip_closest[0],
        "clip_cosine_similarity": clip_closest[1]
    })

# Save results
pd.DataFrame(results).to_csv("worst_similarity_analysis.csv", index=False)
print("Saved comparison results to worst_similarity_analysis.csv")
