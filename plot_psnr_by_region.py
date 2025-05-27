
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your PSNR data CSV
csv_file = "psnr_by_region_summary.csv"  # <-- change this to your actual filename if needed
df = pd.read_csv(csv_file)

# Melt data to long-form for seaborn
melted = df.melt(id_vars=["image_name"], 
                 var_name="noise_region", 
                 value_name="psnr")

# Plot grouped boxplot to see spread and mean per region
plt.figure(figsize=(12, 6))
sns.boxplot(data=melted, x="noise_region", y="psnr", palette="Set2")
plt.title("PSNR Distribution by Noise Region")
plt.xlabel("Noise Region")
plt.ylabel("PSNR Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
