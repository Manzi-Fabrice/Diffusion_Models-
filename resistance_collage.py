import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont

# -------- SETTINGS --------
csv_path = "worse_fragile_overlapping_memorized_images.csv"
base_dir = "recon_vis"
noise_steps = [330, 380, 400, 420, 500, 600]  # Starting from 330
output_file = "fragile_overlapping_memorized_images_labeled.png"
cell_size = (128, 128)  # Resize all images to this (width, height)
padding = 5
label_width = 100  # space for image name on left

# -------- STEP 1: Load sorted image list --------
df = pd.read_csv(csv_path)
image_names = df["image_name"].tolist()

# -------- STEP 2: Load images --------
rows = []
labels = []

for image_name in image_names:
    row_images = []

    base_filename = image_name.replace(".jpg", "")
    original_path = os.path.join(base_dir, "noise_330", f"{base_filename}_original.png")

    if os.path.exists(original_path):
        original_img = Image.open(original_path).resize(cell_size)
    else:
        print(f"Missing original for {base_filename}")
        continue
    row_images.append(original_img)

    for step in noise_steps:
        recon_path = os.path.join(base_dir, f"noise_{step}", f"{base_filename}_reconstructed.png")
        if os.path.exists(recon_path):
            recon_img = Image.open(recon_path).resize(cell_size)
        else:
            print(f"Missing reconstructed image for {base_filename} at step {step}")
            recon_img = Image.new("RGB", cell_size, color="gray")
        row_images.append(recon_img)

    rows.append(row_images)
    labels.append(image_name)

# -------- STEP 3: Stitch rows into a single image with labels --------
cols = len(noise_steps) + 1
collage_width = label_width + cols * (cell_size[0] + padding)
collage_height = len(rows) * (cell_size[1] + padding)
collage = Image.new("RGB", (collage_width, collage_height), color="white")
draw = ImageDraw.Draw(collage)

# Optional: specify a default font
try:
    font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()

for row_idx, (row, label) in enumerate(zip(rows, labels)):
    y = row_idx * (cell_size[1] + padding)
    # Draw label
    draw.text((5, y + cell_size[1] // 3), label, fill="black", font=font)
    # Draw image sequence
    for col_idx, img in enumerate(row):
        x = label_width + col_idx * (cell_size[0] + padding)
        collage.paste(img, (x, y))

# -------- STEP 4: Save --------
collage.save(output_file)
print(f"✅ Saved labeled collage as {output_file}")
