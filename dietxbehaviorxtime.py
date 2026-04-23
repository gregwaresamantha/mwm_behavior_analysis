import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
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

print("\n================ ORIGINAL COLUMN CHECK ================\n")
print("MWM1 columns:")
print(mwm1.columns.tolist())
print("\nMWM2 columns:")
print(mwm2.columns.tolist())

# =====================================================
# 3. STANDARDIZE MWM2 COLUMN NAMES TO MATCH MWM1
# =====================================================
mwm2 = mwm2.rename(columns={
    "Distance (m)": "Distance",
    "Duration (s)": "Duration",
    "Mean speed (m/s)": "MeanSpeed",
    "Trial": "Row",

    "NE : time (s)": "NE_Time",
    "NE : distance (m)": "NE_Distance",
    "NE : average speed (m/s)": "NE_AverageSpeed",

    "NW : time (s)": "NW_Time",
    "NW : distance (m)": "NW_Distance",
    "NW : average speed (m/s)": "NW_AverageSpeed",

    "SE : time (s)": "SE_Time",
    "SE : distance (m)": "SE_Distance",
    "SE : average speed (m/s)": "SE_AverageSpeed",

    "SW : time (s)": "SW_Time",
    "SW : distance (m)": "SW_Distance",
    "SW : average speed (m/s)": "SW_AverageSpeed",

    "Island : distance to first entry (m)": "Island_DistanceToFirstEntry",
    "Island : latency to first entry (s)": "Island_LatencyToFirstEntry",

    "Thigmitaxis : time (s)": "Thigmitaxis_Time",
    "Thigmitaxis : distance (m)": "Thigmitaxis_Distance",
    "Thigmitaxis : average speed (m/s)": "Thigmitaxis_AverageSpeed",

    "Age (months)": "Age_mastersheet"
})

print("\n================ COLUMN CHECK AFTER RENAMING ================\n")
cols1 = set(mwm1.columns)
cols2 = set(mwm2.columns)
print("Columns only in MWM1:", sorted(cols1 - cols2))
print("Columns only in MWM2:", sorted(cols2 - cols1))

# =====================================================
# 4. KEEP ONLY SHARED COLUMNS
# =====================================================
common_cols = sorted(list(set(mwm1.columns) & set(mwm2.columns)))
mwm1 = mwm1[common_cols].copy()
mwm2 = mwm2[common_cols].copy()

print("\nNumber of shared columns kept:", len(common_cols))
print("Shared columns:")
print(common_cols)

# =====================================================
# 5. SCALE CONTROL + HFD DISTANCE DATA (x10)
# NOTE: not needed for time, but harmless to keep for consistency
# =====================================================
distance_cols = [
    "Distance",
    "NE_Distance",
    "NW_Distance",
    "SE_Distance",
    "SW_Distance",
    "Island_DistanceToFirstEntry",
    "Thigmitaxis_Distance"
]

for col in distance_cols:
    if col in mwm1.columns:
        mwm1[col] = mwm1[col] * 10

print("\nScaled Control/HFD distances by x10")

# =====================================================
# 6. CLEAN IDS
# =====================================================
ID_COL = "AnimalID"

if ID_COL not in mwm1.columns or ID_COL not in mwm2.columns:
    raise KeyError(f"'{ID_COL}' not found in one or both behavior files.")

mwm1[ID_COL] = mwm1[ID_COL].astype(str).str.strip()
mwm2[ID_COL] = mwm2[ID_COL].astype(str).str.strip()

# =====================================================
# 7. ASSIGN GROUP LABELS
# =====================================================
mwm1["SourceFile"] = "MWM1012024"
mwm2["SourceFile"] = "MWM12302025"

mwm1["Group"] = mwm1["Diet"].astype(str).str.strip()
mwm2["Group"] = "GLP1"

if "Sex" in mwm1.columns:
    mwm1["Sex_final"] = mwm1["Sex"].astype(str).str.strip()
else:
    mwm1["Sex_final"] = np.nan

if "Sex" in mwm2.columns:
    mwm2["Sex_final"] = mwm2["Sex"].astype(str).str.strip()
else:
    mwm2["Sex_final"] = np.nan

if "Genotype" in mwm1.columns:
    mwm1["Genotype_final"] = mwm1["Genotype"].astype(str).str.strip()
else:
    mwm1["Genotype_final"] = np.nan

if "Genotype" in mwm2.columns:
    mwm2["Genotype_final"] = mwm2["Genotype"].astype(str).str.strip()
else:
    mwm2["Genotype_final"] = np.nan

# =====================================================
# 8. COMBINE FILES
# =====================================================
mwm = pd.concat([mwm1, mwm2], ignore_index=True, sort=False)

for col in ["Group", "Sex_final", "Genotype_final"]:
    mwm[col] = mwm[col].astype(str).str.strip()
    mwm.loc[mwm[col].isin(["nan", "None", ""]), col] = np.nan

print("\n================ COMBINED BEHAVIOR CHECK ================\n")
print("Combined rows:", len(mwm))
print("Unique animals:", mwm[ID_COL].nunique())

print("\nGroup counts (rows):")
print(mwm["Group"].value_counts(dropna=False))

print("\nGroup counts (animals):")
print(mwm[[ID_COL, "Group"]].drop_duplicates()["Group"].value_counts(dropna=False))

# =====================================================
# 9. EXTRACT DAY NUMBER
# =====================================================
df = mwm.copy()

day_source = None
for c in ["Stage", "Test"]:
    if c in df.columns:
        extracted = df[c].astype(str).str.extract(r"(\d+)")[0]
        if extracted.notna().sum() > 0:
            day_source = c
            df["Day_num"] = extracted.astype(float)
            break

if day_source is None:
    raise KeyError("Could not extract day number from Stage or Test.")

print("\nDay number extracted from:", day_source)

# =====================================================
# 10. CREATE TIME-BASED STRATEGY VARIABLE
#     SW time / total trial time
# =====================================================
if "SW_Time" not in df.columns:
    raise KeyError("SW_Time column not found.")
if "Duration" not in df.columns:
    raise KeyError("Duration column not found.")

df["SW_over_total_time"] = df["SW_Time"] / df["Duration"]
df.loc[np.isinf(df["SW_over_total_time"]), "SW_over_total_time"] = np.nan

# =====================================================
# 11. KEEP LEARNING DAYS 1-5
# =====================================================
learning_df = df[df["Day_num"].between(1, 5, inclusive="both")].copy()
learning_df = learning_df.dropna(subset=["SW_over_total_time", "Day_num", "Group", "Sex_final", ID_COL])

print("\n================ LEARNING DATA CHECK ================\n")
print("Rows in learning_df:", len(learning_df))
print("Animals in learning_df:", learning_df[ID_COL].nunique())

print("\nGroup counts (learning rows):")
print(learning_df["Group"].value_counts(dropna=False))

print("\nGroup counts (learning animals):")
print(learning_df[[ID_COL, "Group"]].drop_duplicates()["Group"].value_counts(dropna=False))

# =====================================================
# 12. SET GROUP ORDER + COLORS
# =====================================================
group_order = ["Control", "HFD", "GLP1"]
group_colors = {
    "Control": "#8A2BE2",  # purple
    "HFD": "#FF2D2D",      # red
    "GLP1": "#66E61A"      # green
}

present_groups = [g for g in group_order if g in learning_df["Group"].dropna().unique()]
learning_df["Group"] = pd.Categorical(learning_df["Group"], categories=present_groups, ordered=True)

# =====================================================
# 13. MIXED MODEL: TIME-BASED LEARNING CURVE
# =====================================================
print("\n================ MIXED MODEL: GROUP x DAY (TIME) ================\n")
mixed_formula = "SW_over_total_time ~ Group * Day_num"
mixed_model = smf.mixedlm(
    mixed_formula,
    data=learning_df,
    groups=learning_df[ID_COL]
)
mixed_result = mixed_model.fit(method="lbfgs", reml=False)
print(mixed_result.summary())

# =====================================================
# 14. OPTIONAL FULL MODEL
# =====================================================
fuller_df = learning_df.dropna(subset=["Genotype_final"]).copy()

if len(fuller_df) > 0:
    fuller_df["Group"] = pd.Categorical(fuller_df["Group"], categories=present_groups, ordered=True)
    print("\n================ FULL MIXED MODEL (TIME) ================\n")
    full_formula = "SW_over_total_time ~ Genotype_final * Sex_final * Group * Day_num"
    full_model = smf.mixedlm(
        full_formula,
        data=fuller_df,
        groups=fuller_df[ID_COL]
    )
    full_result = full_model.fit(method="lbfgs", reml=False)
    print(full_result.summary())

# =====================================================
# 15. PER-ANIMAL TIME-BASED LEARNING SLOPE
# =====================================================
def animal_slope(subdf):
    subdf = subdf.dropna(subset=["Day_num", "SW_over_total_time"]).sort_values("Day_num")
    if subdf["Day_num"].nunique() < 2:
        return np.nan
    x = subdf["Day_num"].values
    y = subdf["SW_over_total_time"].values
    return np.polyfit(x, y, 1)[0]

slopes_df = (
    learning_df.groupby(ID_COL, observed=False)
    .apply(animal_slope)
    .reset_index(name="learning_slope_time")
)

animal_info = learning_df[[ID_COL, "Group", "Sex_final", "Genotype_final"]].drop_duplicates(subset=[ID_COL]).copy()
slopes_df = pd.merge(slopes_df, animal_info, on=ID_COL, how="left")
slopes_df = slopes_df.dropna(subset=["learning_slope_time"])
slopes_df["Group"] = pd.Categorical(slopes_df["Group"], categories=present_groups, ordered=True)

print("\n================ PER-ANIMAL TIME SLOPES CHECK ================\n")
print("Animals with slope:", len(slopes_df))
print(slopes_df.head())

print("\nPer-animal group counts:")
print(slopes_df["Group"].value_counts(dropna=False))

# =====================================================
# 16. LINEAR MODEL ON TIME SLOPES
# =====================================================
print("\n================ LINEAR MODEL: learning_slope_time ~ Group ================\n")
slope_lm = smf.ols("learning_slope_time ~ Group", data=slopes_df).fit()
print(slope_lm.summary())

print("\n================ TUKEY POST-HOC FOR GROUP (TIME) ================\n")
tukey = pairwise_tukeyhsd(
    endog=slopes_df["learning_slope_time"],
    groups=slopes_df["Group"].astype(str)
)
print(tukey)

# =====================================================
# 17. GLP1-ONLY MODEL
# =====================================================
glp1_only = slopes_df[slopes_df["Group"] == "GLP1"].copy()

if len(glp1_only) > 0 and glp1_only["Genotype_final"].notna().sum() > 0:
    print("\n================ GLP1-ONLY MODEL (TIME) ================\n")
    glp1_lm = smf.ols("learning_slope_time ~ Genotype_final * Sex_final", data=glp1_only).fit()
    print(glp1_lm.summary())
else:
    print("\nNo GLP1-only genotype model run. Check whether genotype exists in the GLP1 file.")

# =====================================================
# 18. SUMMARY TABLE FOR PLOTTING
# =====================================================
plot_summary = (
    learning_df.groupby(["Group", "Day_num"], observed=False)
    .agg(
        mean_sw_time=("SW_over_total_time", "mean"),
        sem_sw_time=("SW_over_total_time", "sem"),
        mean_sw_raw_time=("SW_Time", "mean"),
        sem_sw_raw_time=("SW_Time", "sem")
    )
    .reset_index()
)

plot_summary = plot_summary.sort_values(["Group", "Day_num"])

print("\n================ PLOT SUMMARY PREVIEW ================\n")
print(plot_summary.head(10))

# =====================================================
# 19. PLOT: SW TIME / TOTAL TIME OVER DAYS
# =====================================================
plt.figure(figsize=(8, 5.5))

for group in present_groups:
    sub = plot_summary[plot_summary["Group"] == group]
    plt.errorbar(
        sub["Day_num"],
        sub["mean_sw_time"],
        yerr=sub["sem_sw_time"],
        marker="o",
        markersize=6,
        linewidth=2,
        capsize=3,
        color=group_colors[group],
        label=group
    )

plt.xlabel("Day")
plt.ylabel("SW time / Total time")
plt.title("Spatial strategy by group (time-based)")
plt.xticks([1, 2, 3, 4, 5], ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"])
plt.legend()
plt.tight_layout()
plt.show()

# =====================================================
# 20. OPTIONAL PLOT: RAW SW TIME OVER DAYS
# =====================================================
plt.figure(figsize=(8, 5.5))

for group in present_groups:
    sub = plot_summary[plot_summary["Group"] == group]
    plt.errorbar(
        sub["Day_num"],
        sub["mean_sw_raw_time"],
        yerr=sub["sem_sw_raw_time"],
        marker="o",
        markersize=6,
        linewidth=2,
        capsize=3,
        color=group_colors[group],
        label=group
    )

plt.xlabel("Day")
plt.ylabel("SW time (s)")
plt.title("Southwest quadrant time by group")
plt.xticks([1, 2, 3, 4, 5], ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"])
plt.legend()
plt.tight_layout()
plt.show()

# =====================================================
# 21. OPTIONAL TABLES FOR EXPORT
# =====================================================
mixed_df = pd.DataFrame({
    "Effect": mixed_result.params.index,
    "Coefficient": mixed_result.params.values,
    "SE": mixed_result.bse.values,
    "p-value": mixed_result.pvalues.values
})
print("\nMixed model table:")
print(mixed_df)

tukey_df = pd.DataFrame(
    data=tukey._results_table.data[1:],
    columns=tukey._results_table.data[0]
)
print("\nTukey table:")
print(tukey_df)

# mixed_df.to_csv("learning_time_mixed_model_table.csv", index=False)
# tukey_df.to_csv("learning_time_tukey_table.csv", index=False)
# slopes_df.to_csv("learning_time_slopes_per_animal.csv", index=False)

import statsmodels.formula.api as smf

print("\n================ MIXED MODEL: TIME =================\n")

learning_df["Group"] = learning_df["Group"].astype("category")

mixed_model_time = smf.mixedlm(
    "SW_over_total_time ~ Group * Day_num",
    data=learning_df,
    groups=learning_df["AnimalID"]
)

mixed_result_time = mixed_model_time.fit(method="lbfgs", reml=False)
print(mixed_result_time.summary())

import numpy as np

def animal_slope_time(subdf):
    subdf = subdf.dropna(subset=["Day_num", "SW_over_total_time"]).sort_values("Day_num")
    if subdf["Day_num"].nunique() < 2:
        return np.nan
    x = subdf["Day_num"].values
    y = subdf["SW_over_total_time"].values
    return np.polyfit(x, y, 1)[0]

slopes_time = (
    learning_df.groupby("AnimalID")
    .apply(animal_slope_time)
    .reset_index(name="learning_slope_time")
)

group_info = learning_df[["AnimalID", "Group"]].drop_duplicates()

slopes_time = slopes_time.merge(group_info, on="AnimalID", how="left")
slopes_time = slopes_time.dropna(subset=["learning_slope_time"])

print("\n================ TIME SLOPES =================\n")
print(slopes_time.head())
print(slopes_time["Group"].value_counts())

print("\n================ LINEAR MODEL: TIME SLOPES =================\n")

slope_lm_time = smf.ols("learning_slope_time ~ Group", data=slopes_time).fit()
print(slope_lm_time.summary())

from statsmodels.stats.multicomp import pairwise_tukeyhsd

print("\n================ TUKEY: TIME SLOPES =================\n")

tukey_time = pairwise_tukeyhsd(
    endog=slopes_time["learning_slope_time"],
    groups=slopes_time["Group"]
)

print(tukey_time)

mixed_time_df = pd.DataFrame({
    "Effect": mixed_result_time.params.index,
    "Coefficient": mixed_result_time.params.values,
    "SE": mixed_result_time.bse.values,
    "p-value": mixed_result_time.pvalues.values
})

print("\nMixed model table:")
print(mixed_time_df)

mixed_time_df.to_csv("time_mixed_model_table.csv", index=False)

tukey_time_df = pd.DataFrame(
    data=tukey_time._results_table.data[1:],
    columns=tukey_time._results_table.data[0]
)

print("\nTukey table:")
print(tukey_time_df)

tukey_time_df.to_csv("time_tukey_table.csv", index=False)

slope_time_df = pd.DataFrame({
    "Effect": slope_lm_time.params.index,
    "Coefficient": slope_lm_time.params.values,
    "SE": slope_lm_time.bse.values,
    "p-value": slope_lm_time.pvalues.values
})

print("\nSlope model table:")
print(slope_time_df)

slope_time_df.to_csv("time_slope_model_table.csv", index=False)
