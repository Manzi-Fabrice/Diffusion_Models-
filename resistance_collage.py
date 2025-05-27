import pandas as pd
import os
from PIL import Image

# -------- SETTINGS --------
csv_path = "top_resistant_images.csv"
base_dir = "recon_vis"
noise_steps = [330, 380, 400, 420, 500, 600]  # Starting from 330
output_file = "resistant_collage.png"
cell_size = (128, 128)  # Resize all images to this (width, height)
padding = 5

# -------- STEP 1: Load sorted image list --------
df = pd.read_csv(csv_path)
image_names = df["image_name"].tolist()

# -------- STEP 2: Load images --------
rows = []
for image_name in image_names:
    row_images = []

    # Use the exact filename (with .jpg)
    base_filename = os.path.splitext(image_name)[0]  # strips the .jpg

    # Load original image from noise_330
    original_path = os.path.join(base_dir, "noise_330", f"{base_filename}_original.png")
    if os.path.exists(original_path):
        original_img = Image.open(original_path).resize(cell_size)
    else:
        print(f"Missing original for {base_filename}")
        continue
    row_images.append(original_img)

    # Load reconstructed images from all noise steps
    for step in noise_steps:
        recon_path = os.path.join(base_dir, f"noise_{step}", f"{base_filename}_reconstructed.png")
        if os.path.exists(recon_path):
            recon_img = Image.open(recon_path).resize(cell_size)
        else:
            print(f"Missing reconstructed image for {base_filename} at step {step}")
            recon_img = Image.new("RGB", cell_size, color="gray")
        row_images.append(recon_img)

    rows.append(row_images)

# -------- STEP 3: Stitch rows into a single image --------
cols = len(noise_steps) + 1  # original + reconstructions
collage_width = cols * (cell_size[0] + padding)
collage_height = len(rows) * (cell_size[1] + padding)

collage = Image.new("RGB", (collage_width, collage_height), color="white")

for row_idx, row in enumerate(rows):
    for col_idx, img in enumerate(row):
        x = col_idx * (cell_size[0] + padding)
        y = row_idx * (cell_size[1] + padding)
        collage.paste(img, (x, y))

# -------- STEP 4: Save --------
collage.save(output_file)
print(f"Saved collage as {output_file}")
