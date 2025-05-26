import pandas as pd

# Load your CSV (assuming it's named 'merged_output.csv')
df = pd.read_csv("merged_output.csv")

# List of corruption parts to check
parts = ["top", "bottom", "left", "right", "center", "center_clear"]

# Create a DataFrame to collect violations
worse_than_all_noise = pd.DataFrame()

# Check for each part
for part in parts:
    mask = df[part] > df["all_noise"]
    if mask.any():
        temp = df[mask].copy()
        temp["worse_part"] = part
        temp["worse_value"] = temp[part]
        temp["difference"] = temp[part] - temp["all_noise"]
        worse_than_all_noise = pd.concat([worse_than_all_noise, temp], ignore_index=True)

# Sort by the difference (descending: biggest violations first)
sorted_df = worse_than_all_noise.sort_values(by="difference", ascending=False)

# Save the result to a CSV file
sorted_df[["image_name", "worse_part", "worse_value", "all_noise", "difference"]].to_csv("worse_than_all_noise_sorted.csv", index=False)

print("Saved sorted results to 'worse_than_all_noise_sorted.csv'")
