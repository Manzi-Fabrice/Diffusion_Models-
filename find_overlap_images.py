import pandas as pd

# -------- SETTINGS --------
csv_a = "worse_than_all_noise_sorted.csv"      # File with local drop images
csv_b = "top_fragile_images.csv"         # File with globally resistant images
output_file = "worse_fragile_overlapping_memorized_images.csv"

# -------- LOAD DATA --------
df_a = pd.read_csv(csv_a)
df_b = pd.read_csv(csv_b)

# -------- FIND OVERLAP --------
set_a = set(df_a["image_name"])
set_b = set(df_b["image_name"])
intersecting = sorted(set_a.intersection(set_b))

# -------- SAVE RESULTS --------
overlap_df = pd.DataFrame(intersecting, columns=["image_name"])
overlap_df.to_csv(output_file, index=False)

print(f"✅ Found {len(overlap_df)} overlapping images.")
print(f"Saved to: {output_file}")
