"""
control_group_baseline_analysis.py
Merge MWM data with CSV metadata and test baseline/control-group comparisons.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# -----------------------------
# SETTINGS
# -----------------------------
MWM_FILE = "mwm_data.csv"
META_FILE = "metadata.csv"   # <-- rename your CSV to this

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

sns.set_style("whitegrid")

# -----------------------------
# HELPERS
# -----------------------------
def normalize_cols(df):
    df.columns = [
        str(c).strip().replace(" ", "_").replace("/", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def find_first_matching(cols, candidates):
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def infer_group_from_row(row):
    joined = " ".join([str(v) for v in row.values if pd.notna(v)]).upper()

    if "CVN" in joined:
        return "CVN"
    if "HN" in joined and "APOE" not in joined:
        return "HN control"
    if "APOE22" in joined:
        return "APOE22HN"
    if "APOE33" in joined:
        return "APOE33HN"
    if "APOE44" in joined:
        return "APOE44HN"

    return np.nan


# -----------------------------
# LOAD MWM
# -----------------------------
mwm = pd.read_csv(MWM_FILE)
mwm.columns = mwm.columns.str.strip()

mwm["Animal Code"] = mwm["Animal Code"].astype(str).str.strip()

for col in [
    "Distance Probe Day 5",
    "Distance Probe Day 8",
    "SW Distance Probe Day 5",
    "SW Distance Probe Day 8",
]:
    mwm[col] = pd.to_numeric(mwm[col], errors="coerce")

mwm["day5_SW_TOTAL"] = mwm["SW Distance Probe Day 5"] / mwm["Distance Probe Day 5"]
mwm["day8_SW_TOTAL"] = mwm["SW Distance Probe Day 8"] / mwm["Distance Probe Day 8"]


# -----------------------------
# LOAD METADATA CSV
# -----------------------------
meta = pd.read_csv(META_FILE)
meta = normalize_cols(meta)
meta["source_sheet"] = "csv"

print("Metadata columns:", list(meta.columns))

# find ID column automatically
id_col = find_first_matching(
    meta.columns,
    ["Animal_ID", "AnimalCode", "Animal_Code", "Animal Code", "ID"]
)

if id_col is None:
    raise ValueError("Could not find animal ID column — check metadata column names above.")

meta["Animal_ID"] = meta[id_col].astype(str).str.strip()

# assign groups
meta["baseline_group"] = meta.apply(infer_group_from_row, axis=1)

meta.to_csv(RESULTS_DIR / "metadata_all_sheets_combined.csv", index=False)


# -----------------------------
# MERGE
# -----------------------------
merged = mwm.merge(
    meta[["Animal_ID", "baseline_group"]].drop_duplicates(),
    left_on="Animal Code",
    right_on="Animal_ID",
    how="left"
)

merged.to_csv(RESULTS_DIR / "mwm_with_metadata_controls.csv", index=False)


# -----------------------------
# FILTER
# -----------------------------
control_df = merged.dropna(subset=["baseline_group"]).copy()

if control_df.empty:
    print("No control groups detected — check metadata labels.")
    exit()


# =============================
# DAY 5 MODEL
# =============================
d5 = control_df.dropna(subset=["day5_SW_TOTAL", "Distance Probe Day 5"]).copy()

if d5["baseline_group"].nunique() >= 2:

    model5 = smf.ols(
        "day5_SW_TOTAL ~ C(baseline_group) + Q('Distance Probe Day 5')",
        data=d5
    ).fit()

    out5 = model5.summary2().tables[1]
    out5.to_csv(RESULTS_DIR / "control_day5_glm_results.csv")

    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(data=d5, x="baseline_group", y="day5_SW_TOTAL", cut=0)

    sns.stripplot(
        data=d5,
        x="baseline_group",
        y="day5_SW_TOTAL",
        color="black",
        alpha=0.7,
        jitter=0.1
    )

    plt.title("Probe Day 5 by Baseline Group")
    plt.ylabel("SW Distance / Total Distance")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "control_day5.png", dpi=300)
    plt.close()


# =============================
# DAY 8 MODEL
# =============================
d8 = control_df.dropna(subset=["day8_SW_TOTAL", "Distance Probe Day 8"]).copy()

if d8["baseline_group"].nunique() >= 2:

    model8 = smf.ols(
        "day8_SW_TOTAL ~ C(baseline_group) + Q('Distance Probe Day 8')",
        data=d8
    ).fit()

    out8 = model8.summary2().tables[1]
    out8.to_csv(RESULTS_DIR / "control_day8_glm_results.csv")

    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(data=d8, x="baseline_group", y="day8_SW_TOTAL", cut=0)

    sns.stripplot(
        data=d8,
        x="baseline_group",
        y="day8_SW_TOTAL",
        color="black",
        alpha=0.7,
        jitter=0.1
    )

    plt.title("Probe Day 8 by Baseline Group")
    plt.ylabel("SW Distance / Total Distance")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "control_day8.png", dpi=300)
    plt.close()


print("DONE ✅ Check results/ and figures/")