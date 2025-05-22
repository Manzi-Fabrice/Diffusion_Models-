import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
top_df = pd.read_csv("top_2_percent_similarity_analysis.csv")
bottom_df = pd.read_csv("worst_similarity_analysis.csv")

# Combine for seaborn
top_df["group"] = "Top 1% (High Fidelity)"
bottom_df["group"] = "Bottom 1% (Low Fidelity)"
df_combined = pd.concat([top_df, bottom_df])

# Set style
sns.set(style="whitegrid")

# ---- pHash Distance ----
plt.figure(figsize=(10, 4))
sns.kdeplot(data=df_combined, x="phash_distance", hue="group", fill=True, common_norm=False, alpha=0.5, linewidth=2)
plt.title("pHash Distance Distribution")
plt.xlabel("pHash Distance")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("phash_distance_density.png")
plt.close()

# ---- pHash Boxplot ----
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_combined, x="group", y="phash_distance", palette={"Top 1% (High Fidelity)": "green", "Bottom 1% (Low Fidelity)": "red"})
plt.title("pHash Distance Spread")
plt.tight_layout()
plt.savefig("phash_distance_boxplot.png")
plt.close()

# ---- CLIP Cosine Similarity ----
plt.figure(figsize=(10, 4))
sns.kdeplot(data=df_combined, x="clip_cosine_similarity", hue="group", fill=True, common_norm=False, alpha=0.5, linewidth=2)
plt.title("CLIP Cosine Similarity Distribution")
plt.xlabel("Cosine Similarity")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("clip_similarity_density.png")
plt.close()

# ---- CLIP Boxplot ----
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_combined, x="group", y="clip_cosine_similarity", palette={"Top 1% (High Fidelity)": "blue", "Bottom 1% (Low Fidelity)": "orange"})
plt.title("CLIP Cosine Similarity Spread")
plt.tight_layout()
plt.savefig("clip_similarity_boxplot.png")
plt.close()
