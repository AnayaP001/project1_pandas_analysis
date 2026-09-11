"""
Data Analysis and Visualization with Pandas & Matplotlib
Dataset: Iris flower dataset (iris.csv)

This script:
 1. Loads a CSV file with Pandas
 2. Performs basic statistical analysis
 3. Creates a bar chart, scatter plot, and heatmap with Matplotlib/Seaborn
 4. Prints insights derived from the analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. LOAD THE CSV FILE
# ---------------------------------------------------------
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "iris.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("STEP 1: DATA OVERVIEW")
print("=" * 60)
print(f"Shape of dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")
print("First 5 rows:")
print(df.head(), "\n")
print("Column data types:")
print(df.dtypes, "\n")
print("Missing values per column:")
print(df.isnull().sum(), "\n")

# ---------------------------------------------------------
# 2. BASIC ANALYSIS
# ---------------------------------------------------------
print("=" * 60)
print("STEP 2: BASIC STATISTICAL ANALYSIS")
print("=" * 60)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# Overall summary statistics
print("Descriptive statistics (all numeric columns):")
print(df[numeric_cols].describe(), "\n")

# Average of a selected column
selected_col = "petal length (cm)"
avg_value = df[selected_col].mean()
print(f"Average of '{selected_col}': {avg_value:.2f} cm\n")

# Average of each numeric column, grouped by species
group_means = df.groupby("species")[numeric_cols].mean()
print("Average of each measurement, grouped by species:")
print(group_means, "\n")

# Correlation matrix (used later for the heatmap)
corr_matrix = df[numeric_cols].corr()
print("Correlation matrix:")
print(corr_matrix, "\n")

# ---------------------------------------------------------
# 3. VISUALIZATIONS
# ---------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

# --- 3a. BAR CHART: average petal length per species ---
fig, ax = plt.subplots(figsize=(7, 5))
colors = ["#4C72B0", "#DD8452", "#55A868"]
bars = ax.bar(group_means.index, group_means[selected_col], color=colors)
ax.set_title(f"Average {selected_col.title()} by Species", fontsize=14, weight="bold")
ax.set_xlabel("Species")
ax.set_ylabel(f"{selected_col.title()}")
for bar in bars:
    height = bar.get_height()
    ax.annotate(f"{height:.2f}", (bar.get_x() + bar.get_width() / 2, height),
                textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bar_chart.png"), dpi=150)
plt.close()

# --- 3b. SCATTER PLOT: petal length vs petal width, colored by species ---
fig, ax = plt.subplots(figsize=(7, 5))
species_colors = {"setosa": "#4C72B0", "versicolor": "#DD8452", "virginica": "#55A868"}
for sp, color in species_colors.items():
    subset = df[df["species"] == sp]
    ax.scatter(subset["petal length (cm)"], subset["petal width (cm)"],
               label=sp, color=color, alpha=0.75, edgecolor="white", s=60)
ax.set_title("Petal Length vs Petal Width by Species", fontsize=14, weight="bold")
ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.legend(title="Species")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "scatter_plot.png"), dpi=150)
plt.close()

# --- 3c. HEATMAP: correlation matrix of numeric features ---
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
ax.set_yticklabels(numeric_cols)

# annotate each cell with the correlation value
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center", va="center",
                color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")

ax.set_title("Correlation Heatmap of Iris Features", fontsize=14, weight="bold")
fig.colorbar(im, ax=ax, label="Correlation Coefficient")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "heatmap.png"), dpi=150)
plt.close()

print("Saved: bar_chart.png, scatter_plot.png, heatmap.png")

# ---------------------------------------------------------
# 4. INSIGHTS
# ---------------------------------------------------------
strongest_pair = corr_matrix.where(~np.eye(len(corr_matrix), dtype=bool)).abs().stack().idxmax()
strongest_val = corr_matrix.loc[strongest_pair]

print("\n" + "=" * 60)
print("STEP 4: KEY INSIGHTS")
print("=" * 60)
print(f"""
1. Sample size: {df.shape[0]} flowers, evenly split across 3 species
   ({df['species'].value_counts().to_dict()}).

2. The average '{selected_col}' across all species is {avg_value:.2f} cm,
   but this hides big differences between species (see grouped means above) —
   setosa has much smaller petals than versicolor and virginica.

3. Species separate very clearly by petal measurements: the bar chart shows
   petal length roughly doubles from setosa to virginica, and the scatter
   plot shows the three species occupy distinct, mostly non-overlapping
   regions of petal length/width space. This makes petal size a strong
   predictor of species.

4. The heatmap shows petal length and petal width are very strongly
   correlated (r = {corr_matrix.loc['petal length (cm)','petal width (cm)']:.2f}),
   and petal length correlates strongly with sepal length
   (r = {corr_matrix.loc['petal length (cm)','sepal length (cm)']:.2f}) too.
   Sepal width is the odd one out — it's weakly/negatively correlated with
   the other features (r = {corr_matrix.loc['sepal width (cm)','petal length (cm)']:.2f}
   with petal length), meaning it doesn't move in step with overall flower size.

5. Strongest overall relationship: {strongest_pair[0]} vs {strongest_pair[1]}
   (r = {strongest_val:.2f}).
""")
