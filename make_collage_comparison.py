
import os
from PIL import Image
import math

image_list_file = "combined_image_list.txt"
noise_steps = [300, 350, 400, 450, 500]
original_dir = "originals"
recons_base = "recons"

with open(image_list_file, "r") as f:
    image_list = [line.strip() for line in f if line.strip()]

rows = len(image_list)
cols = 1 + len(noise_steps)

sample_img = Image.open(os.path.join(original_dir, image_list[0]))
img_width, img_height = sample_img.size
collage = Image.new("RGB", (cols * img_width, rows * img_height), color=(255, 255, 255))

for row_idx, fname in enumerate(image_list):
    try:
        orig = Image.open(os.path.join(original_dir, fname)).convert("RGB")
        collage.paste(orig, (0, row_idx * img_height))
        for step_idx, step in enumerate(noise_steps):
            recon_path = os.path.join(recons_base, str(step), fname)
            if os.path.exists(recon_path):
                recon = Image.open(recon_path).convert("RGB")
                collage.paste(recon, ((step_idx + 1) * img_width, row_idx * img_height))
    except Exception as e:
        print(f"Error for {fname}: {e}")

collage.save("progressive_collage.jpg")
print("✅ Saved comparison collage as progressive_collage.jpg")
