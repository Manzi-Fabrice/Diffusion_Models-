import pandas as pd

# Load the CSV file
file_path = "worse_than_all_noise_sorted.csv"
df = pd.read_csv(file_path)

# Filter to keep only rows where 'difference' > 1
df_filtered = df[df['difference'] > 1]

# Drop duplicates based on 'image_name', keeping the first occurrence
df_unique = df_filtered.drop_duplicates(subset='image_name', keep='first')

# Overwrite the original file with the filtered data
df_unique.to_csv(file_path, index=False)

print(f"Filtered and saved file with {len(df_unique)} unique entries.")
