"""
Created on 3/27/26
@author: samgregware
"""


"""
mass_by_genotype_analysis.py
Stratified delta-mass analyses within genotype for learning and memory.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# -----------------------------
# SETTINGS
# -----------------------------
MWM_FILE = "mwm_data.csv"
WEIGHT_FILE = "mouse_body_weights_sema.csv"

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

GENO_LABELS = {
    "APOE22HN": "APOE ε2",
    "APOE33HN": "APOE ε3",
    "APOE44HN": "APOE ε4",
}
GENO_ORDER = ["APOE ε2", "APOE ε3", "APOE ε4"]

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.title_fontsize": 14,
    "legend.fontsize": 12,
})

# -----------------------------
# LOAD WEIGHTS
# -----------------------------
weights = pd.read_csv(WEIGHT_FILE)
weights.columns = weights.columns.str.strip()
weights = weights[["Animal_ID", "AnimalWeight_g_100825", "AnimalWeight_g_121025"]].copy()
weights["Animal_ID"] = weights["Animal_ID"].astype(str).str.strip()
weights["AnimalWeight_g_100825"] = pd.to_numeric(weights["AnimalWeight_g_100825"], errors="coerce")
weights["AnimalWeight_g_121025"] = pd.to_numeric(weights["AnimalWeight_g_121025"], errors="coerce")
weights = weights.dropna().copy()
weights["delta_mass"] = (
    (weights["AnimalWeight_g_121025"] - weights["AnimalWeight_g_100825"])
    / weights["AnimalWeight_g_100825"]
) * 100

# -----------------------------
# LOAD MWM
# -----------------------------
mwm = pd.read_csv(MWM_FILE)
mwm.columns = mwm.columns.str.strip()
mwm["Animal Code"] = mwm["Animal Code"].astype(str).str.strip()
mwm["Genotype"] = mwm["Genotype"].astype(str).str.strip()
mwm["genotype_label"] = mwm["Genotype"].map(GENO_LABELS)

# ratios for learning
for col in [
    "Day1_SW", "Day2_SW", "Day3_SW", "Day4_SW", "Day5_SW",
    "Day1_Totaldistance", "Day2_Totaldistance", "Day3_Totaldistance",
    "Day4_Totaldistance", "Day5_Totaldistance",
    "Distance Probe Day 5", "Distance Probe Day 8",
    "SW Distance Probe Day 5", "SW Distance Probe Day 8",
]:
    mwm[col] = pd.to_numeric(mwm[col], errors="coerce")

for d in range(1, 6):
    mwm[f"Day{d}_ratio"] = mwm[f"Day{d}_SW"] / mwm[f"Day{d}_Totaldistance"]

days = np.array([1, 2, 3, 4, 5], dtype=float)
ratio_cols = [f"Day{d}_ratio" for d in range(1, 6)]

def compute_ratio_slope(row):
    y = row[ratio_cols].values.astype(float)
    if np.isnan(y).any():
        return np.nan
    return np.polyfit(days, y, 1)[0]

mwm["learning_ratio_slope"] = mwm.apply(compute_ratio_slope, axis=1)
mwm["day5_SW_TOTAL"] = mwm["SW Distance Probe Day 5"] / mwm["Distance Probe Day 5"]
mwm["day8_SW_TOTAL"] = mwm["SW Distance Probe Day 8"] / mwm["Distance Probe Day 8"]
mwm["delta_memory"] = mwm["day8_SW_TOTAL"] - mwm["day5_SW_TOTAL"]

behavior = mwm[[
    "Animal Code", "Genotype", "genotype_label",
    "learning_ratio_slope", "day5_SW_TOTAL", "day8_SW_TOTAL", "delta_memory"
]].copy().rename(columns={"Animal Code": "Animal_ID"})

merged = weights.merge(behavior, on="Animal_ID", how="inner")
merged.to_csv(RESULTS_DIR / "mass_by_genotype_merged.csv", index=False)

# -----------------------------
# MODEL + PLOT HELPERS
# -----------------------------
def fit_within_genotype(df, outcome):
    rows = []
    for geno in GENO_ORDER:
        sub = df[df["genotype_label"] == geno].dropna(subset=["delta_mass", outcome]).copy()
        if len(sub) < 3:
            continue
        model = smf.ols(f"{outcome} ~ delta_mass", data=sub).fit()
        beta = model.params.get("delta_mass", np.nan)
        p = model.pvalues.get("delta_mass", np.nan)
        r2 = model.rsquared
        rows.append({
            "genotype_label": geno,
            "outcome": outcome,
            "beta_delta_mass": beta,
            "p_value_delta_mass": p,
            "r_squared": r2,
            "N": int(model.nobs)
        })
    return pd.DataFrame(rows)

def add_panel_stats(ax, stats_df, geno):
    sub = stats_df[stats_df["genotype_label"] == geno]
    if len(sub) == 0:
        return
    row = sub.iloc[0]
    text = (
        f"β = {row['beta_delta_mass']:.3f}\n"
        f"p = {row['p_value_delta_mass']:.3g}\n"
        f"R² = {row['r_squared']:.3f}\n"
        f"N = {int(row['N'])}"
    )
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

def make_facet_regplot(df, outcome, y_label, filename, title):
    stats_df = fit_within_genotype(df, outcome)

    g = sns.FacetGrid(
        df, col="genotype_label", col_order=GENO_ORDER,
        sharex=True, sharey=True, height=4.8, aspect=0.95
    )
    g.map_dataframe(sns.regplot, x="delta_mass", y=outcome, scatter_kws={"s": 55})

    for ax, geno in zip(g.axes.flat, GENO_ORDER):
        add_panel_stats(ax, stats_df, geno)
        ax.set_title(geno, fontweight="bold")
        ax.set_xlabel("Percent Body Weight Change (%)")
        ax.set_ylabel(y_label)

    g.figure.suptitle(title, y=1.02, fontsize=18)
    g.figure.tight_layout()
    g.figure.savefig(FIG_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(g.figure)

    return stats_df

# -----------------------------
# RUN STRATIFIED ANALYSES
# -----------------------------
all_stats = []

for outcome, ylabel, fname, title in [
    ("day5_SW_TOTAL", "Probe Day 5 SW/TOTAL", "delta_mass_vs_probe5_by_genotype.png", "Delta Mass vs Probe Day 5 by Genotype"),
    ("day8_SW_TOTAL", "Probe Day 8 SW/TOTAL", "delta_mass_vs_probe8_by_genotype.png", "Delta Mass vs Probe Day 8 by Genotype"),
    ("delta_memory", "Δ Memory (Day 8 - Day 5)", "delta_mass_vs_delta_memory_by_genotype.png", "Delta Mass vs Change in Memory by Genotype"),
    ("learning_ratio_slope", "Learning Slope (SW/Total, Days 1-5)", "delta_mass_vs_learning_slope_by_genotype.png", "Delta Mass vs Learning Slope by Genotype"),
]:
    stats_df = make_facet_regplot(merged, outcome, ylabel, fname, title)
    all_stats.append(stats_df)

all_stats_df = pd.concat(all_stats, ignore_index=True)
all_stats_df.to_csv(RESULTS_DIR / "mass_by_genotype_regression_results.csv", index=False)

print("Saved:")
print("- results/mass_by_genotype_merged.csv")
print("- results/mass_by_genotype_regression_results.csv")
print("- figures/delta_mass_vs_probe5_by_genotype.png")
print("- figures/delta_mass_vs_probe8_by_genotype.png")
print("- figures/delta_mass_vs_delta_memory_by_genotype.png")
print("- figures/delta_mass_vs_learning_slope_by_genotype.png")



if __name__ == '__main__':
    pass
