# Data Analysis & Visualization with Pandas and Matplotlib

## Overview
This project loads a CSV dataset with **Pandas**, performs basic statistical
analysis (means, group-by aggregations, correlations), and produces three
visualizations with **Matplotlib**: a bar chart, a scatter plot, and a
correlation heatmap. It closes with a short written interpretation of the
results.

## Dataset
`data/iris.csv` — the classic Iris flower dataset (150 samples, 3 species,
4 numeric measurements: sepal length/width, petal length/width). It's a
standard, well-known dataset for demonstrating this kind of analysis.

To use a different dataset instead, replace `data/iris.csv` with your own
CSV and update the column names referenced in `src/analysis.py`
(`selected_col`, `numeric_cols`, and the `groupby` key).

## Project Structure
```
project1_pandas_analysis/
├── data/
│   └── iris.csv              # input dataset
├── src/
│   └── analysis.py           # standalone script version
├── notebooks/
│   └── analysis.ipynb        # notebook version (same analysis, cell-by-cell)
├── outputs/
│   ├── bar_chart.png
│   ├── scatter_plot.png
│   └── heatmap.png
├── requirements.txt
└── README.md
```

## How to Run

**Script version:**
```bash
cd src
python3 analysis.py
```
Charts are written to `../outputs/`.

**Notebook version:**
```bash
jupyter notebook notebooks/analysis.ipynb
```
Run all cells top to bottom. Charts are written to `../outputs/` relative
to the notebook.

## Setup
```bash
pip install -r requirements.txt
```

## What It Does
1. **Load & inspect** — shape, dtypes, missing values, first rows
2. **Basic analysis** — `.describe()`, average of a selected column,
   per-species (`groupby`) averages, correlation matrix
3. **Visualizations**
   - Bar chart: average petal length by species
   - Scatter plot: petal length vs. petal width, colored by species
   - Heatmap: correlation matrix of all four numeric features
4. **Insights** — printed summary of the key patterns found (species
   separate cleanly on petal size; petal length/width are strongly
   correlated; sepal width behaves differently from the other features)

## Key Findings
- Petal length and petal width are very strongly correlated (r ≈ 0.96).
- The three species occupy clearly distinct regions in petal
  length/width space, making petal size a strong predictor of species.
- Sepal width is only weakly (and slightly negatively) correlated with the
  other features, unlike the rest of the measurements which move together.
