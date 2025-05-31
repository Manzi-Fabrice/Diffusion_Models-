import torch
import clip
from PIL import Image
import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load your list of 7 overlapping images
image_list = pd.read_csv("worse_fragile_overlapping_memorized_images.csv")["image_name"].tolist()
base_path = "recon_vis/noise_330"  # Path to original images

embeddings = []
valid_names = []

for img_name in image_list:
    # Strip the .jpg extension before constructing the filename
    base_name = img_name.replace(".jpg", "")
    img_path = os.path.join(base_path, f"{base_name}_original.png")

    if os.path.exists(img_path):
        image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image).squeeze().cpu().numpy()
        embeddings.append(embedding)
        valid_names.append(img_name)
    else:
        print(f"Missing: {img_path}")

# Reduce dimensions to 2D for plotting
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c='purple', s=100)
for i, name in enumerate(valid_names):
    plt.text(reduced[i, 0] + 0.01, reduced[i, 1] + 0.01, name, fontsize=9)
plt.title("2D PCA of Embeddings for Overlapping Fragile Images")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("embedding_2d_visualization.png")

