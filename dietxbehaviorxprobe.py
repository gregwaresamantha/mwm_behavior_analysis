import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# =====================================================
# 1. LOAD FILES
# =====================================================
mwm1 = pd.read_csv("MWM1012024.csv")   # Control + HFD
mwm2 = pd.read_csv("MWM12302025.csv")  # GLP1 cohort

# =====================================================
# 2. CLEAN COLUMN NAMES
# =====================================================
for df in [mwm1, mwm2]:
    df.columns = df.columns.str.strip()

# =====================================================
# 3. STANDARDIZE MWM2 COLUMN NAMES
# =====================================================
mwm2 = mwm2.rename(columns={
    "Distance (m)": "Distance",
    "Duration (s)": "Duration",
    "Mean speed (m/s)": "MeanSpeed",
    "Trial": "Row",
    "SW : time (s)": "SW_Time",
    "SW : distance (m)": "SW_Distance",
    "NE : time (s)": "NE_Time",
    "NW : time (s)": "NW_Time",
    "SE : time (s)": "SE_Time"
})

# =====================================================
# 4. KEEP ONLY SHARED COLUMNS
# =====================================================
common_cols = list(set(mwm1.columns) & set(mwm2.columns))
mwm1 = mwm1[common_cols].copy()
mwm2 = mwm2[common_cols].copy()

# =====================================================
# 5. SCALE CONTROL + HFD DISTANCE (x10)
# =====================================================
if "Distance" in mwm1.columns:
    mwm1["Distance"] *= 10
if "SW_Distance" in mwm1.columns:
    mwm1["SW_Distance"] *= 10

# =====================================================
# 6. ASSIGN GROUPS
# =====================================================
mwm1["Group"] = mwm1["Diet"].astype(str).str.strip()
mwm2["Group"] = "GLP1"

mwm1["AnimalID"] = mwm1["AnimalID"].astype(str)
mwm2["AnimalID"] = mwm2["AnimalID"].astype(str)

# =====================================================
# 7. MERGE DATASETS
# =====================================================
df = pd.concat([mwm1, mwm2], ignore_index=True)

# =====================================================
# 8. EXTRACT DAY NUMBER
# =====================================================
df["Day_num"] = df["Stage"].astype(str).str.extract(r"(\d+)")[0].astype(float)

# =====================================================
# 9. CREATE METRIC
# =====================================================
df["SW_over_total_dist"] = df["SW_Distance"] / df["Distance"]
df.replace([np.inf, -np.inf], np.nan, inplace=True)

YVAR = "SW_over_total_dist"
YLABEL = "SW distance / Total distance"

# =====================================================
# 10. FIND PROBE ROWS (ROBUST)
# =====================================================
stage_str = df["Stage"].astype(str)
test_str = df["Test"].astype(str)

probe_mask = (
    stage_str.str.contains("probe", case=False, na=False) |
    test_str.str.contains("probe", case=False, na=False)
)

probe_df = df[probe_mask].copy()
probe_df = probe_df[probe_df["Day_num"].isin([5, 8])].copy()

print("\nRows after probe filter:", len(probe_df))

if probe_df.empty:
    print("No probe rows found — check your Stage/Test labels.")
    raise SystemExit

# =====================================================
# 11. AGGREGATE PER ANIMAL
# =====================================================
probe_animal = (
    probe_df.groupby(["AnimalID", "Group", "Day_num"], as_index=False)
    .agg(y=(YVAR, "mean"))
    .dropna()
)

print("\nProbe counts:")
print(probe_animal.groupby(["Group", "Day_num"]).size())

# =====================================================
# 12. WITHIN GROUP STATS
# =====================================================
print("\n=== WITHIN GROUP STATS ===")

for group in probe_animal["Group"].unique():
    sub = probe_animal[probe_animal["Group"] == group]
    wide = sub.pivot(index="AnimalID", columns="Day_num", values="y").dropna()

    if 5 in wide.columns and 8 in wide.columns and len(wide) > 1:
        t, p = ttest_rel(wide[5], wide[8])
        print(f"{group}: p = {p:.4f}")

# =====================================================
# 13. ACROSS GROUP STATS
# =====================================================
print("\n=== ACROSS GROUP STATS ===")

for day in [5, 8]:
    sub = probe_animal[probe_animal["Day_num"] == day]

    if sub["Group"].nunique() >= 2:
        groups = [sub[sub["Group"] == g]["y"].dropna() for g in sub["Group"].unique()]

        if all(len(g) > 1 for g in groups):
            f, p = f_oneway(*groups)
            print(f"Day {day} ANOVA p = {p:.4f}")

            tukey = pairwise_tukeyhsd(sub["y"], sub["Group"])
            print(tukey)

# =====================================================
# 14. FINAL VIOLIN PLOT (NO DOTS)
# =====================================================
group_order = ["Control", "HFD", "GLP1"]
present_groups = [g for g in group_order if g in probe_animal["Group"].unique()]

day_colors = {
    5: "#63B34E",  # green
    8: "#5B9BD5"   # blue
}

fig, axes = plt.subplots(1, len(present_groups), figsize=(5.5 * len(present_groups), 5.5), sharey=True)

if len(present_groups) == 1:
    axes = [axes]

for ax, group in zip(axes, present_groups):

    sub = probe_animal[probe_animal["Group"] == group]

    vals5 = sub[sub["Day_num"] == 5]["y"].dropna().values
    vals8 = sub[sub["Day_num"] == 8]["y"].dropna().values

    if len(vals5) == 0 or len(vals8) == 0:
        continue

    # VIOLIN
    v = ax.violinplot([vals5, vals8], positions=[1, 2], showextrema=False)

    v["bodies"][0].set_facecolor(day_colors[5])
    v["bodies"][1].set_facecolor(day_colors[8])

    for b in v["bodies"]:
        b.set_edgecolor("black")
        b.set_alpha(0.9)

    # BOX
    ax.boxplot(
        [vals5, vals8],
        positions=[1, 2],
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="none", color="black"),
        medianprops=dict(color="black")
    )

    ax.set_title(group, fontsize=16)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Day 5", "Day 8"])

axes[0].set_ylabel(YLABEL)
plt.suptitle("Probe: Day 5 vs Day 8 by Group", fontsize=20)
plt.tight_layout()
plt.show()