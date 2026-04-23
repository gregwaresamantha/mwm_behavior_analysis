"""
Created on 3/27/26
@author: samgregware
"""

"""
probe_within_day_analysis.py
Run within-day probe comparisons and day-by-genotype interaction models
for MWM memory performance (SW / Total Distance).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import ttest_rel

# -----------------------------
# SETTINGS
# -----------------------------
DATA_FILE = "mwm_data.csv"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

GENO_REF = "APOE33HN"
SEX_REF = "F"

GENO_LABELS = {
"APOE22HN": "APOE ε2",
"APOE33HN": "APOE ε3",
"APOE44HN": "APOE ε4",
}
GENO_ORDER = ["APOE ε2", "APOE ε3", "APOE ε4"]
DAY_COLORS = {"Day 5": "#55B748", "Day 8": "#4C97D7"}

sns.set_style("white")
plt.rcParams.update({
"font.size": 12,
"axes.titlesize": 18,
"axes.labelsize": 16,
"legend.title_fontsize": 14,
"legend.fontsize": 12,
})

# -----------------------------
# LOAD + CLEAN
# -----------------------------
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

df["Animal Code"] = df["Animal Code"].astype(str).str.strip()
df["Genotype"] = df["Genotype"].astype(str).str.strip()
df["Sex"] = df["Sex"].astype(str).str.strip()
df["genotype_label"] = df["Genotype"].map(GENO_LABELS)

for col in ["Distance Probe Day 5", "Distance Probe Day 8",
"SW Distance Probe Day 5", "SW Distance Probe Day 8"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["day5_SW_TOTAL"] = df["SW Distance Probe Day 5"] / df["Distance Probe Day 5"]
df["day8_SW_TOTAL"] = df["SW Distance Probe Day 8"] / df["Distance Probe Day 8"]

probe = df[[
"Animal Code", "Genotype", "Sex", "genotype_label",
"day5_SW_TOTAL", "day8_SW_TOTAL"
]].copy()

# -----------------------------
# HELPERS
# -----------------------------
def export_glm(result, label, outfile, n):
    out = pd.DataFrame({
    "term": result.params.index,
    "estimate": result.params.values,
    "std_error": result.bse.values,
    "t_value": result.tvalues.values,
    "p_value": result.pvalues.values,
    })
    out["model"] = label
    out["N"] = n
    out.to_csv(RESULTS_DIR / outfile, index=False)
    return out

def export_mixedlm(result, label, outfile, n_mice, n_obs):
    out = pd.DataFrame({
    "term": result.params.index,
    "estimate": result.params.values,
    "std_error": result.bse.values,
    "z_value": result.tvalues.values,
    "p_value": result.pvalues.values,
    })
    out["model"] = label
    out["N_mice"] = n_mice
    out["N_obs"] = n_obs
    out.to_csv(RESULTS_DIR / outfile, index=False)
    return out

def get_p(model, term):
    return float(model.pvalues[term]) if term in model.pvalues.index else np.nan

def compute_box_stats(df_long, value_col, measure_name):
    stats = (
    df_long.groupby(["genotype_label", "Day"])[value_col]
    .agg(
    Q1=lambda x: x.quantile(0.25),
    Median="median",
    Q3=lambda x: x.quantile(0.75),
    Mean="mean",
    SD="std",
    N="count"
)
.reset_index()
)
    stats["measure"] = measure_name
    return stats

# -----------------------------
# DAY 5 GLM
# -----------------------------
day5_df = probe.dropna(subset=["day5_SW_TOTAL"]).copy()

day5_formula = (
f"day5_SW_TOTAL ~ "
f"C(Genotype, Treatment(reference='{GENO_REF}')) + "
f"C(Sex, Treatment(reference='{SEX_REF}'))"
)

day5_glm = smf.ols(day5_formula, data=day5_df).fit()
export_glm(day5_glm, f"GLM: {day5_formula}", "probe_day5_glm_results.csv", len(day5_df))

day5_p_22 = get_p(day5_glm, f"C(Genotype, Treatment(reference='{GENO_REF}'))[T.APOE22HN]")
day5_p_44 = get_p(day5_glm, f"C(Genotype, Treatment(reference='{GENO_REF}'))[T.APOE44HN]")

# -----------------------------
# DAY 8 GLM
# -----------------------------
day8_df = probe.dropna(subset=["day8_SW_TOTAL"]).copy()

day8_formula = (
f"day8_SW_TOTAL ~ "
f"C(Genotype, Treatment(reference='{GENO_REF}')) + "
f"C(Sex, Treatment(reference='{SEX_REF}'))"
)

day8_glm = smf.ols(day8_formula, data=day8_df).fit()
export_glm(day8_glm, f"GLM: {day8_formula}", "probe_day8_glm_results.csv", len(day8_df))

day8_p_22 = get_p(day8_glm, f"C(Genotype, Treatment(reference='{GENO_REF}'))[T.APOE22HN]")
day8_p_44 = get_p(day8_glm, f"C(Genotype, Treatment(reference='{GENO_REF}'))[T.APOE44HN]")

# -----------------------------
# LONG FORMAT + MIXED MODEL
# -----------------------------
probe_long = probe.melt(
id_vars=["Animal Code", "Genotype", "Sex", "genotype_label"],
value_vars=["day5_SW_TOTAL", "day8_SW_TOTAL"],
var_name="Day",
value_name="SW_ratio"
)

probe_long["Day"] = probe_long["Day"].map({
"day5_SW_TOTAL": "Day 5",
"day8_SW_TOTAL": "Day 8",
})
probe_long["Day_num"] = probe_long["Day"].map({"Day 5": 5, "Day 8": 8})
probe_long["Day_c"] = probe_long["Day_num"] - probe_long["Day_num"].mean()
probe_long["SW_ratio"] = pd.to_numeric(probe_long["SW_ratio"], errors="coerce")
probe_long = probe_long.dropna(subset=["SW_ratio"]).copy()
probe_long["MouseID"] = probe_long["Animal Code"].astype("category")

mixed_formula = (
f"SW_ratio ~ "
f"C(Day, Treatment(reference='Day 5')) * C(Genotype, Treatment(reference='{GENO_REF}')) + "
f"C(Day, Treatment(reference='Day 5')) * C(Sex, Treatment(reference='{SEX_REF}'))"
)

probe_mixed = smf.mixedlm(
mixed_formula,
data=probe_long,
groups=probe_long["MouseID"]
).fit(reml=False)

export_mixedlm(
probe_mixed,
f"MixedLM: {mixed_formula} + (1|MouseID)",
"probe_day5_day8_mixedlm_results.csv",
probe_long["MouseID"].nunique(),
probe_long.shape[0]
)

day_p = get_p(probe_mixed, "C(Day, Treatment(reference='Day 5'))[T.Day 8]")
int22_p = get_p(
probe_mixed,
f"C(Day, Treatment(reference='Day 5'))[T.Day 8]:C(Genotype, Treatment(reference='{GENO_REF}'))[T.APOE22HN]"
)
int44_p = get_p(
probe_mixed,
f"C(Day, Treatment(reference='Day 5'))[T.Day 8]:C(Genotype, Treatment(reference='{GENO_REF}'))[T.APOE44HN]"
)

# -----------------------------
# BOXPLOT STATS CSV
# -----------------------------
box_stats = compute_box_stats(probe_long, "SW_ratio", "Probe SW/Total")
box_stats.to_csv(RESULTS_DIR / "probe_day5_day8_boxplot_stats.csv", index=False)

# -----------------------------
# OVERALL DAY5 vs DAY8 PLOT
# -----------------------------
plt.figure(figsize=(7, 6))
ax = sns.violinplot(
data=probe_long, x="Day", y="SW_ratio", hue="Day",
palette=DAY_COLORS, inner=None, cut=0, linewidth=1.6, legend=False
)
sns.boxplot(
data=probe_long, x="Day", y="SW_ratio", width=0.15,
showcaps=True,
boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.6},
whiskerprops={"linewidth": 1.6},
medianprops={"color": "black", "linewidth": 2},
showfliers=False, ax=ax
)
sns.stripplot(
data=probe_long, x="Day", y="SW_ratio",
color="black", alpha=0.7, jitter=0.08, ax=ax
)

stats_text = (
f"Mixed model\n"
f"Day 8 vs Day 5 p = {day_p:.3g}\n"
f"Day×ε2 vs ε3 p = {int22_p:.3g}\n"
f"Day×ε4 vs ε3 p = {int44_p:.3g}"
)
ax.text(
0.02, 0.98, stats_text,
transform=ax.transAxes, ha="left", va="top",
bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

d5 = probe_long.loc[probe_long["Day"] == "Day 5", "SW_ratio"].dropna()
d8 = probe_long.loc[probe_long["Day"] == "Day 8", "SW_ratio"].dropna()
quart_text = (
f"Day 5: Q1={d5.quantile(0.25):.3f}, Med={d5.median():.3f}, Q3={d5.quantile(0.75):.3f}\n"
f"Day 8: Q1={d8.quantile(0.25):.3f}, Med={d8.median():.3f}, Q3={d8.quantile(0.75):.3f}"
)
ax.text(
0.98, 0.98, quart_text,
transform=ax.transAxes, ha="right", va="top", fontsize=10,
bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

ax.set_title("Probe Memory: Day 5 vs Day 8")
ax.set_xlabel("")
ax.set_ylabel("SW Distance / Total Distance")
plt.tight_layout()
plt.savefig(FIG_DIR / "probe_day5_vs_day8_overall.png", dpi=300)
plt.close()

# -----------------------------
# DAY 5 BY GENOTYPE PLOT
# -----------------------------
plt.figure(figsize=(8, 6))
ax = sns.violinplot(
data=day5_df, x="genotype_label", y="day5_SW_TOTAL",
order=GENO_ORDER, inner=None, cut=0, linewidth=1.6
)
sns.boxplot(
data=day5_df, x="genotype_label", y="day5_SW_TOTAL",
order=GENO_ORDER, width=0.15, showcaps=True,
boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.6},
whiskerprops={"linewidth": 1.6},
medianprops={"color": "black", "linewidth": 2},
showfliers=False, ax=ax
)
sns.stripplot(
data=day5_df, x="genotype_label", y="day5_SW_TOTAL",
order=GENO_ORDER, color="black", alpha=0.7, jitter=0.08, ax=ax
)

ax.text(
0.02, 0.98,
f"Day 5 GLM\nε2 vs ε3 p = {day5_p_22:.3g}\nε4 vs ε3 p = {day5_p_44:.3g}",
transform=ax.transAxes, ha="left", va="top",
bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)
ax.set_title("Probe Day 5 by Genotype")
ax.set_xlabel("")
ax.set_ylabel("SW Distance / Total Distance")
plt.tight_layout()
plt.savefig(FIG_DIR / "probe_day5_by_genotype.png", dpi=300)
plt.close()

# -----------------------------
# DAY 8 BY GENOTYPE PLOT
# -----------------------------
plt.figure(figsize=(8, 6))
ax = sns.violinplot(
data=day8_df, x="genotype_label", y="day8_SW_TOTAL",
order=GENO_ORDER, inner=None, cut=0, linewidth=1.6
)
sns.boxplot(
data=day8_df, x="genotype_label", y="day8_SW_TOTAL",
order=GENO_ORDER, width=0.15, showcaps=True,
boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.6},
whiskerprops={"linewidth": 1.6},
medianprops={"color": "black", "linewidth": 2},
showfliers=False, ax=ax
)
sns.stripplot(
data=day8_df, x="genotype_label", y="day8_SW_TOTAL",
order=GENO_ORDER, color="black", alpha=0.7, jitter=0.08, ax=ax
)

ax.text(
0.02, 0.98,
f"Day 8 GLM\nε2 vs ε3 p = {day8_p_22:.3g}\nε4 vs ε3 p = {day8_p_44:.3g}",
transform=ax.transAxes, ha="left", va="top",
bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)
ax.set_title("Probe Day 8 by Genotype")
ax.set_xlabel("")
ax.set_ylabel("SW Distance / Total Distance")
plt.tight_layout()
plt.savefig(FIG_DIR / "probe_day8_by_genotype.png", dpi=300)
plt.close()

# -----------------------------
# DAY5 vs DAY8 BY GENOTYPE
# -----------------------------
g = sns.catplot(
data=probe_long, x="Day", y="SW_ratio", hue="Day", col="genotype_label",
col_order=GENO_ORDER, kind="violin", inner=None, palette=DAY_COLORS,
cut=0, linewidth=1.6, height=5.2, aspect=0.85, legend=False, sharey=True
)

for ax, geno in zip(g.axes.flat, GENO_ORDER):
    sub = probe_long[probe_long["genotype_label"] == geno]

    sns.boxplot(
    data=sub, x="Day", y="SW_ratio", width=0.15, showcaps=True,
    boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},
    whiskerprops={"linewidth": 1.5},
    medianprops={"color": "black", "linewidth": 2},
    showfliers=False, ax=ax
)
sns.stripplot(
data=sub, x="Day", y="SW_ratio",
color="black", alpha=0.7, jitter=0.08, ax=ax
)

d5_pair = sub[sub["Day"] == "Day 5"][["Animal Code", "SW_ratio"]].rename(columns={"SW_ratio": "d5"})
d8_pair = sub[sub["Day"] == "Day 8"][["Animal Code", "SW_ratio"]].rename(columns={"SW_ratio": "d8"})
paired = d5_pair.merge(d8_pair, on="Animal Code", how="inner")

paired_p = np.nan
if len(paired) >= 2:
    _, paired_p = ttest_rel(paired["d5"], paired["d8"])

text = (
f"Paired p = {paired_p:.3g}\n"
f"Day5: {paired['d5'].quantile(0.25):.2f} | {paired['d5'].median():.2f} | {paired['d5'].quantile(0.75):.2f}\n"
f"Day8: {paired['d8'].quantile(0.25):.2f} | {paired['d8'].median():.2f} | {paired['d8'].quantile(0.75):.2f}"
)
ax.text(
0.02, 0.98, text,
transform=ax.transAxes, ha="left", va="top", fontsize=9,
bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)
ax.set_title(geno, fontweight="bold")
ax.set_xlabel("")

g.axes.flat[0].text(
0.02, 0.02,
f"Overall mixed model\nDay 8 vs Day 5 p = {day_p:.3g}\nDay×ε2 vs ε3 p = {int22_p:.3g}\nDay×ε4 vs ε3 p = {int44_p:.3g}",
transform=g.axes.flat[0].transAxes, ha="left", va="bottom", fontsize=9,
bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)
g.set_ylabels("SW Distance / Total Distance")
g.figure.suptitle("Probe Memory: Day 5 vs Day 8 by Genotype", y=1.02, fontsize=18)
g.figure.tight_layout()
g.figure.savefig(FIG_DIR / "probe_day5_vs_day8_by_genotype.png", dpi=300, bbox_inches="tight")
plt.close(g.figure)

print("Saved:")
print("- results/probe_day5_glm_results.csv")
print("- results/probe_day8_glm_results.csv")
print("- results/probe_day5_day8_mixedlm_results.csv")
print("- results/probe_day5_day8_boxplot_stats.csv")
print("- figures/probe_day5_by_genotype.png")
print("- figures/probe_day8_by_genotype.png")
print("- figures/probe_day5_vs_day8_overall.png")
print("- figures/probe_day5_vs_day8_by_genotype.png")



