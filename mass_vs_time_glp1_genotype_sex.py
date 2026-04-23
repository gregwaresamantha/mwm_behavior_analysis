import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import ttest_rel
import re

# =====================================================
# SETTINGS
# =====================================================
FILE = "Mouse_HFD_GLP1_WeightChange.csv"

ID_COL = "Animal_ID"
SEX_COL = "Sex"
GENOTYPE_COL = "Genotype"

# colors for plots
GENOTYPE_COLORS = {
    "APOE22HN": "#4C78A8",
    "APOE33HN": "#F58518",
    "APOE44HN": "#54A24B",
    "APOE2": "#4C78A8",
    "APOE3": "#F58518",
    "APOE4": "#54A24B",
}
SEX_COLORS = {
    "M": "#4C78A8",
    "F": "#E45756",
    "male": "#4C78A8",
    "female": "#E45756",
}

# =====================================================
# HELPERS
# =====================================================
def extract_date(colname):
    m = re.search(r"_(\d{6})$", colname)
    return m.group(1) if m else np.nan

def cohens_d_paired(x, y):
    diff = y - x
    sd = diff.std(ddof=1)
    if pd.isna(sd) or sd == 0:
        return np.nan
    return diff.mean() / sd

def summarize_start_end(group_df, id_col, label_col, label_name):
    rows = []
    for label, sub in group_df.groupby(label_col):
        per_mouse = (
            sub.sort_values([id_col, "Date"])
            .groupby(id_col)
            .apply(lambda x: pd.Series({
                "start_weight": x.iloc[0]["Weight_g"],
                "end_weight": x.iloc[-1]["Weight_g"],
                "start_pct": x.iloc[0]["pct_change"],
                "end_pct": x.iloc[-1]["pct_change"],
            }))
            .reset_index()
        )

        if len(per_mouse) > 1:
            t_stat, p_val = ttest_rel(per_mouse["start_weight"], per_mouse["end_weight"])
            d = cohens_d_paired(per_mouse["start_weight"], per_mouse["end_weight"])
        else:
            t_stat, p_val, d = np.nan, np.nan, np.nan

        rows.append({
            label_name: label,
            "n_mice": len(per_mouse),
            "mean_start_weight_g": per_mouse["start_weight"].mean(),
            "mean_end_weight_g": per_mouse["end_weight"].mean(),
            "mean_delta_weight_g": (per_mouse["end_weight"] - per_mouse["start_weight"]).mean(),
            "mean_start_pct": per_mouse["start_pct"].mean(),
            "mean_end_pct": per_mouse["end_pct"].mean(),
            "paired_t": t_stat,
            "paired_p": p_val,
            "cohens_d": d,
        })
    return pd.DataFrame(rows)

def make_summary_by_time(df_long, group_col):
    summary = (
        df_long.groupby([group_col, "Day"], as_index=False)
        .agg(
            mean_pct=("pct_change", "mean"),
            sd_pct=("pct_change", "std"),
            n=("pct_change", "count")
        )
    )
    summary["sem_pct"] = summary["sd_pct"] / np.sqrt(summary["n"])
    return summary

def plot_group_trajectory(summary_df, group_col, title, color_map, outfile):
    groups = summary_df[group_col].dropna().unique().tolist()

    plt.figure(figsize=(8, 5.5))
    for g in groups:
        sub = summary_df[summary_df[group_col] == g].sort_values("Day")
        color = color_map.get(g, None)
        plt.errorbar(
            sub["Day"],
            sub["mean_pct"],
            yerr=sub["sem_pct"],
            marker="o",
            linewidth=2,
            capsize=3,
            label=str(g),
            color=color
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Day")
    plt.ylabel("% Change from Baseline Weight")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.show()

# =====================================================
# 1. LOAD DATA
# =====================================================
df = pd.read_csv(FILE)
df.columns = df.columns.str.strip()

print("Columns:")
print(df.columns.tolist())

# =====================================================
# 2. IDENTIFY WEIGHT COLUMNS
# =====================================================
weight_cols = [c for c in df.columns if c.startswith("AnimalWeight")]

if ID_COL not in df.columns:
    raise KeyError(f"Missing ID column: {ID_COL}")
if SEX_COL not in df.columns:
    raise KeyError(f"Missing sex column: {SEX_COL}")
if GENOTYPE_COL not in df.columns:
    raise KeyError(f"Missing genotype column: {GENOTYPE_COL}")
if not weight_cols:
    raise ValueError("No AnimalWeight columns found.")

print("\nWeight columns found:")
print(weight_cols)

# =====================================================
# 3. RESHAPE WIDE -> LONG
# =====================================================
long_df = df.melt(
    id_vars=[ID_COL, SEX_COL, GENOTYPE_COL],
    value_vars=weight_cols,
    var_name="WeightColumn",
    value_name="Weight_g"
)

long_df[ID_COL] = (
    long_df[ID_COL]
    .astype(str)
    .str.strip()
    .str.replace(".0", "", regex=False)
)

long_df[SEX_COL] = long_df[SEX_COL].astype(str).str.strip()
long_df[GENOTYPE_COL] = long_df[GENOTYPE_COL].astype(str).str.strip()
long_df["Weight_g"] = pd.to_numeric(long_df["Weight_g"], errors="coerce")
long_df = long_df.dropna(subset=["Weight_g"]).copy()

# =====================================================
# 4. EXTRACT DATE
# =====================================================
long_df["DateCode"] = long_df["WeightColumn"].apply(extract_date)
long_df["Date"] = pd.to_datetime(long_df["DateCode"], format="%m%d%y", errors="coerce")
long_df = long_df.dropna(subset=["Date"]).copy()

long_df = long_df.sort_values([ID_COL, "Date"]).copy()
long_df["Day"] = (long_df["Date"] - long_df["Date"].min()).dt.days

# =====================================================
# 5. CALCULATE BASELINE + % CHANGE
# =====================================================
baseline_df = (
    long_df.groupby(ID_COL, as_index=False)
    .first()[[ID_COL, "Weight_g"]]
    .rename(columns={"Weight_g": "baseline"})
)

long_df = pd.merge(long_df, baseline_df, on=ID_COL, how="left")
long_df["pct_change"] = ((long_df["Weight_g"] - long_df["baseline"]) / long_df["baseline"]) * 100

print("\nPreview:")
print(long_df.head())

# =====================================================
# 6. GENOTYPE ANALYSIS
# =====================================================
print("\n================ GENOTYPE ANALYSIS ================\n")

geno_df = long_df.copy()
geno_df[GENOTYPE_COL] = geno_df[GENOTYPE_COL].astype("category")
geno_df[ID_COL] = geno_df[ID_COL].astype("category")
geno_df["day_c"] = geno_df["Day"] - geno_df["Day"].mean()

geno_model = smf.mixedlm(
    f"pct_change ~ day_c * {GENOTYPE_COL}",
    data=geno_df,
    groups=geno_df[ID_COL]
)
geno_result = geno_model.fit(method="lbfgs", reml=False)

print(geno_result.summary())

# start vs end by genotype
geno_start_end = summarize_start_end(geno_df, ID_COL, GENOTYPE_COL, "Genotype")
print("\nStart vs end by genotype:")
print(geno_start_end)

# trajectory summary + plot
geno_summary = make_summary_by_time(geno_df, GENOTYPE_COL)
plot_group_trajectory(
    geno_summary,
    GENOTYPE_COL,
    "GLP1 Cohort: % Weight Change Over Time by Genotype",
    GENOTYPE_COLORS,
    "glp1_pct_change_by_genotype.png"
)

# =====================================================
# 7. SEX ANALYSIS
# =====================================================
print("\n================ SEX ANALYSIS ================\n")

sex_df = long_df.copy()
sex_df[SEX_COL] = sex_df[SEX_COL].astype("category")
sex_df[ID_COL] = sex_df[ID_COL].astype("category")
sex_df["day_c"] = sex_df["Day"] - sex_df["Day"].mean()

sex_model = smf.mixedlm(
    f"pct_change ~ day_c * {SEX_COL}",
    data=sex_df,
    groups=sex_df[ID_COL]
)
sex_result = sex_model.fit(method="lbfgs", reml=False)

print(sex_result.summary())

# start vs end by sex
sex_start_end = summarize_start_end(sex_df, ID_COL, SEX_COL, "Sex")
print("\nStart vs end by sex:")
print(sex_start_end)

# trajectory summary + plot
sex_summary = make_summary_by_time(sex_df, SEX_COL)
plot_group_trajectory(
    sex_summary,
    SEX_COL,
    "GLP1 Cohort: % Weight Change Over Time by Sex",
    SEX_COLORS,
    "glp1_pct_change_by_sex.png"
)

# =====================================================
# 8. SAVE OUTPUT TABLES
# =====================================================
geno_summary.to_csv("glp1_pct_change_summary_by_genotype.csv", index=False)
sex_summary.to_csv("glp1_pct_change_summary_by_sex.csv", index=False)
geno_start_end.to_csv("glp1_start_end_by_genotype.csv", index=False)
sex_start_end.to_csv("glp1_start_end_by_sex.csv", index=False)

# save coefficient tables
geno_coef = pd.DataFrame({
    "Effect": geno_result.params.index,
    "Coefficient": geno_result.params.values,
    "SE": geno_result.bse.values,
    "p_value": geno_result.pvalues.values
})
sex_coef = pd.DataFrame({
    "Effect": sex_result.params.index,
    "Coefficient": sex_result.params.values,
    "SE": sex_result.bse.values,
    "p_value": sex_result.pvalues.values
})

geno_coef.to_csv("glp1_mixed_model_by_genotype.csv", index=False)
sex_coef.to_csv("glp1_mixed_model_by_sex.csv", index=False)

print("\nSaved:")
print("- glp1_pct_change_by_genotype.png")
print("- glp1_pct_change_by_sex.png")
print("- glp1_pct_change_summary_by_genotype.csv")
print("- glp1_pct_change_summary_by_sex.csv")
print("- glp1_start_end_by_genotype.csv")
print("- glp1_start_end_by_sex.csv")
print("- glp1_mixed_model_by_genotype.csv")
print("- glp1_mixed_model_by_sex.csv")