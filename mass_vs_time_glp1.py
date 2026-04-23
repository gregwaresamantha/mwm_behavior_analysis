import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import ttest_rel
import re

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("Mouse_HFD_GLP1_WeightChange.csv")
df.columns = df.columns.str.strip()

ID_COL = "Animal_ID"

weight_cols = [c for c in df.columns if c.startswith("AnimalWeight")]

if ID_COL not in df.columns:
    raise KeyError(f"{ID_COL} not found.")

if not weight_cols:
    raise ValueError("No AnimalWeight columns found.")

# ===============================
# RESHAPE WIDE → LONG
# ===============================
long_df = df.melt(
    id_vars=[ID_COL],
    value_vars=weight_cols,
    var_name="WeightColumn",
    value_name="Weight_g"
)

# clean IDs
long_df[ID_COL] = (
    long_df[ID_COL]
    .astype(str)
    .str.strip()
    .str.replace(".0", "", regex=False)
)

# numeric weights
long_df["Weight_g"] = pd.to_numeric(long_df["Weight_g"], errors="coerce")
long_df = long_df.dropna(subset=["Weight_g"]).copy()

# ===============================
# EXTRACT DATE
# ===============================
def extract_date(col):
    m = re.search(r"_(\d{6})$", col)
    return m.group(1) if m else np.nan

long_df["DateCode"] = long_df["WeightColumn"].apply(extract_date)
long_df["Date"] = pd.to_datetime(long_df["DateCode"], format="%m%d%y", errors="coerce")
long_df = long_df.dropna(subset=["Date"]).copy()

# sort
long_df = long_df.sort_values([ID_COL, "Date"]).copy()

# relative day
long_df["Day"] = (long_df["Date"] - long_df["Date"].min()).dt.days

# ===============================
# % CHANGE FROM BASELINE
# ===============================
long_df["baseline"] = long_df.groupby(ID_COL)["Weight_g"].transform("first")

long_df["pct_change"] = (
    (long_df["Weight_g"] - long_df["baseline"]) /
    long_df["baseline"]
) * 100

print("\nPreview:")
print(long_df.head())

# ===============================
# SUMMARY STATS (MEAN ± SEM)
# ===============================
summary_df = (
    long_df.groupby(["Day"], as_index=False)
    .agg(
        mean_pct=("pct_change", "mean"),
        sd_pct=("pct_change", "std"),
        n=("pct_change", "count")
    )
)

summary_df["sem_pct"] = summary_df["sd_pct"] / np.sqrt(summary_df["n"])

print("\nSummary:")
print(summary_df.head())

# ===============================
# PLOT: % CHANGE
# ===============================
plt.figure(figsize=(9, 6))

plt.errorbar(
    summary_df["Day"],
    summary_df["mean_pct"],
    yerr=summary_df["sem_pct"],
    marker="o",
    capsize=3,
    linewidth=2
)

plt.axhline(0, linestyle="--")  # baseline reference

plt.xlabel("Day")
plt.ylabel("% Change from Baseline")
plt.title("GLP1 Cohort Percent Weight Change Over Time")

plt.tight_layout()
plt.show()

# ===============================
# MIXED MODEL (BEST STAT TEST)
# ===============================
mixed_model = smf.mixedlm(
    "pct_change ~ Day",
    data=long_df,
    groups=long_df[ID_COL]
)

mixed_result = mixed_model.fit(method="lbfgs", reml=False)

print("\n=== MIXED MODEL: %Change ~ Day + (1|Animal) ===")
print(mixed_result.summary())

# ===============================
# START VS END (EASY INTERPRETATION)
# ===============================
delta_df = (
    long_df.sort_values([ID_COL, "Date"])
    .groupby(ID_COL)
    .apply(lambda x: pd.Series({
        "start_weight": x.iloc[0]["Weight_g"],
        "end_weight": x.iloc[-1]["Weight_g"],
        "delta_weight": x.iloc[-1]["Weight_g"] - x.iloc[0]["Weight_g"],
        "pct_change": ((x.iloc[-1]["Weight_g"] - x.iloc[0]["Weight_g"]) / x.iloc[0]["Weight_g"]) * 100
    }))
    .reset_index()
)

# paired t-test
t_stat, p_val = ttest_rel(delta_df["start_weight"], delta_df["end_weight"])

# effect size (Cohen's d)
mean_delta = delta_df["delta_weight"].mean()
sd_delta = delta_df["delta_weight"].std()
cohens_d = mean_delta / sd_delta

print("\n=== START VS END ===")
print(f"N = {len(delta_df)}")
print(f"Mean start weight = {delta_df['start_weight'].mean():.3f} g")
print(f"Mean end weight = {delta_df['end_weight'].mean():.3f} g")
print(f"Mean Δweight = {mean_delta:.3f} g")
print(f"Mean % change = {delta_df['pct_change'].mean():.3f} %")
print(f"Paired t-test p = {p_val:.4g}")
print(f"Cohen's d = {cohens_d:.3f}")

# ===============================
# SAVE OUTPUTS
# ===============================
summary_df.to_csv("glp1_percent_change_summary.csv", index=False)
delta_df.to_csv("glp1_percent_change_start_end.csv", index=False)

print("\nSaved:")
print("- glp1_percent_change_summary.csv")
print("- glp1_percent_change_start_end.csv")