import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# =====================================================
# HELPER FUNCTION
# =====================================================
def get_stars(p):
    if pd.isna(p):
        return "ns"
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


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
TITLE = "Probe: Day 5 vs Day 8 by Group"

# =====================================================
# 10. FIND PROBE ROWS
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
    print("No probe rows found. Check Stage/Test labels.")
    raise SystemExit

# =====================================================
# 11. AGGREGATE TO ONE VALUE PER ANIMAL PER DAY
# =====================================================
probe_animal = (
    probe_df.groupby(["AnimalID", "Group", "Day_num"], as_index=False)
    .agg(y=(YVAR, "mean"))
    .dropna()
)

print("\nProbe counts:")
print(probe_animal.groupby(["Group", "Day_num"]).size())

# =====================================================
# 12. WITHIN-GROUP STATS (Day 5 vs Day 8)
# =====================================================
within_results = []
within_pvals = {}

print("\n=== WITHIN GROUP STATS ===")
for group in sorted(probe_animal["Group"].unique()):
    sub = probe_animal[probe_animal["Group"] == group].copy()
    wide = sub.pivot(index="AnimalID", columns="Day_num", values="y").dropna()

    if 5 in wide.columns and 8 in wide.columns and len(wide) > 1:
        t_stat, p_val = ttest_rel(wide[5], wide[8])
        within_pvals[group] = p_val
        within_results.append({
            "Group": group,
            "n_pairs": len(wide),
            "Day5_mean": wide[5].mean(),
            "Day8_mean": wide[8].mean(),
            "t_stat": t_stat,
            "p_value": p_val
        })
        print(f"{group}: p = {p_val:.4g}")
    else:
        within_pvals[group] = np.nan
        within_results.append({
            "Group": group,
            "n_pairs": len(wide),
            "Day5_mean": np.nan,
            "Day8_mean": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan
        })

within_df = pd.DataFrame(within_results)

# =====================================================
# 13. ACROSS-GROUP STATS (ANOVA + TUKEY)
# =====================================================
across_results = []
tukey_tables = {}

print("\n=== ACROSS GROUP STATS ===")
for day in [5, 8]:
    sub = probe_animal[probe_animal["Day_num"] == day].copy()
    groups_present = sorted(sub["Group"].dropna().unique())
    group_arrays = [sub.loc[sub["Group"] == g, "y"].dropna().values for g in groups_present]
    group_arrays = [arr for arr in group_arrays if len(arr) > 1]

    if len(group_arrays) >= 2:
        f_stat, p_val = f_oneway(*group_arrays)
        across_results.append({
            "Day": int(day),
            "F_stat": f_stat,
            "p_value": p_val
        })
        print(f"Day {int(day)} ANOVA p = {p_val:.4g}")

        tukey = pairwise_tukeyhsd(sub["y"], sub["Group"])
        tukey_df = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0]
        )
        tukey_tables[int(day)] = tukey_df
        print(tukey_df)

across_df = pd.DataFrame(across_results)

# =====================================================
# 14. QUARTILE TABLE FOR PANEL ANNOTATIONS
# =====================================================
quartile_rows = []
for group in sorted(probe_animal["Group"].unique()):
    for day in [5, 8]:
        vals = probe_animal.loc[
            (probe_animal["Group"] == group) & (probe_animal["Day_num"] == day),
            "y"
        ].dropna().values

        if len(vals) > 0:
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
        else:
            q1 = med = q3 = np.nan

        quartile_rows.append({
            "Group": group,
            "Day": int(day),
            "Q1": q1,
            "Median": med,
            "Q3": q3,
            "n": len(vals)
        })

quartile_df = pd.DataFrame(quartile_rows)

# =====================================================
# 15. PRINT TABLES
# =====================================================
print("\n=== WITHIN-GROUP TABLE ===")
print(within_df)

print("\n=== ACROSS-GROUP ANOVA TABLE ===")
print(across_df)

print("\n=== QUARTILES TABLE ===")
print(quartile_df)

for day, tdf in tukey_tables.items():
    print(f"\n=== TUKEY DAY {day} ===")
    print(tdf)

# =====================================================
# 16. SAVE TABLES
# =====================================================
within_df.to_csv("probe_within_group_day5_vs_day8.csv", index=False)
across_df.to_csv("probe_across_group_anova.csv", index=False)
quartile_df.to_csv("probe_quartiles_by_group_day.csv", index=False)

for day, tdf in tukey_tables.items():
    tdf.to_csv(f"probe_day{day}_tukey.csv", index=False)

# =====================================================
# 17. FINAL VIOLIN PLOT WITH PANEL TEXT + STARS
# =====================================================
group_order = ["Control", "HFD", "GLP1"]
present_groups = [g for g in group_order if g in probe_animal["Group"].unique()]

if len(present_groups) == 0:
    print("No groups available to plot.")
    raise SystemExit

day_colors = {
    5: "#63B34E",   # green
    8: "#5B9BD5"    # blue
}

fig, axes = plt.subplots(
    1, len(present_groups),
    figsize=(5.6 * len(present_groups), 5.8),
    sharey=True
)

if len(present_groups) == 1:
    axes = [axes]

for ax, group in zip(axes, present_groups):
    sub = probe_animal[probe_animal["Group"] == group].copy()

    vals5 = sub.loc[sub["Day_num"] == 5, "y"].dropna().values
    vals8 = sub.loc[sub["Day_num"] == 8, "y"].dropna().values

    if len(vals5) == 0 or len(vals8) == 0:
        ax.set_visible(False)
        continue

    # calculate range FIRST
    y_max = max(np.max(vals5), np.max(vals8))
    y_min = min(np.min(vals5), np.min(vals8))
    y_range = y_max - y_min if y_max > y_min else 1

    # violin
    v = ax.violinplot(
        [vals5, vals8],
        positions=[1, 2],
        widths=0.8,
        showextrema=False
    )

    v["bodies"][0].set_facecolor(day_colors[5])
    v["bodies"][1].set_facecolor(day_colors[8])

    for body in v["bodies"]:
        body.set_edgecolor("#4a4a4a")
        body.set_alpha(0.95)

    # boxplot
    ax.boxplot(
        [vals5, vals8],
        positions=[1, 2],
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor="none", edgecolor="black", linewidth=1.6),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=1.4),
        capprops=dict(color="black", linewidth=1.4)
    )

    # quartile text
    qsub = quartile_df[quartile_df["Group"] == group].copy()
    q5 = qsub[qsub["Day"] == 5].iloc[0]
    q8 = qsub[qsub["Day"] == 8].iloc[0]

    panel_p = within_pvals.get(group, np.nan)
    p_text = f"Day p = {panel_p:.3g}" if pd.notna(panel_p) else "Day p = NA"

    stats_text = (
        f"{p_text}\n"
        f"Q1 | Med | Q3\n"
        f"Day 5: {q5['Q1']:.2f} | {q5['Median']:.2f} | {q5['Q3']:.2f}\n"
        f"Day 8: {q8['Q1']:.2f} | {q8['Median']:.2f} | {q8['Q3']:.2f}"
    )

    ax.text(
        1.5,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none")
    )

    # significance bracket + stars
    if not np.isnan(panel_p):
        stars = get_stars(panel_p)
        y_star = y_max + 0.03 * y_range

        ax.plot([1, 1, 2, 2],
                [y_star - 0.015 * y_range, y_star, y_star, y_star - 0.015 * y_range],
                color="black", linewidth=1.4)

        ax.text(
            1.5,
            y_star + 0.01 * y_range,
            stars,
            ha="center",
            va="bottom",
            fontsize=14
        )

    ax.set_title(group, fontsize=18, fontweight="bold")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Day 5", "Day 8"], fontsize=12)
    ax.tick_params(axis="y", labelsize=11)

axes[0].set_ylabel(YLABEL, fontsize=16)
plt.suptitle(TITLE, fontsize=22, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print("\nSaved:")
print("- probe_within_group_day5_vs_day8.csv")
print("- probe_across_group_anova.csv")
print("- probe_quartiles_by_group_day.csv")
for day in tukey_tables:
    print(f"- probe_day{day}_tukey.csv")