"""
Created on 3/15/26
@author: samgregware
"""
if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    import statsmodels.formula.api as smf

    # -----------------------------
    # 1. LOAD WEIGHT DATA
    # -----------------------------
    weights = pd.read_csv("mouse_body_weights_sema.csv", header=1)
    weights.columns = weights.columns.str.strip()

    print("Weight columns:")
    print(weights.columns.tolist())


    # keep only useful columns
    weights = weights[[
        "Animal_ID",
        "AnimalWeight_g_112825",
        "AnimalWeight_g_120125",
        "AnimalWeight_g_120825",
        "AnimalWeight_g_121025"
    ]].copy()

    # ensure numeric
    weight_cols = [
        "AnimalWeight_g_112825",
        "AnimalWeight_g_120125",
        "AnimalWeight_g_120825",
        "AnimalWeight_g_121025"
    ]

    for col in weight_cols:
        weights[col] = pd.to_numeric(weights[col], errors="coerce")

    # -----------------------------
    # 2. COMPUTE DELTA WEIGHT
    # -----------------------------
    weights["delta_weight"] = weights["AnimalWeight_g_121025"] - weights[
        "AnimalWeight_g_112825"]

    # -----------------------------
    # 3. COMPUTE WEIGHT SLOPE
    # -----------------------------
    timepoints = np.array([1, 2, 3, 4])


    def compute_slope(row):
        y = row[weight_cols].values.astype(float)
        return np.polyfit(timepoints, y, 1)[0]


    weights["weight_slope"] = weights.apply(compute_slope, axis=1)

    print("\nWeight metrics preview:")
    print(weights[["Animal_ID", "delta_weight", "weight_slope"]].head())

    # -----------------------------
    # 4. LOAD LEARNING DATA
    # -----------------------------
    learning = pd.read_csv("mwm_learning_data.csv")
    learning.columns = learning.columns.str.strip()

    learning = learning[[
        "Animal Code",
        "SW Day 1 Average Distance (m)",
        "SW Day 2 Average Distance (m)",
        "SW Day 3 Average Distance (m)",
        "SW Day 4 Average Distance (m)",
        "SW Day 5 Average Distance (m)"
    ]].copy()

    learning = learning.rename(columns={"Animal Code": "Animal_ID"})

    learning_cols = [
        "SW Day 1 Average Distance (m)",
        "SW Day 2 Average Distance (m)",
        "SW Day 3 Average Distance (m)",
        "SW Day 4 Average Distance (m)",
        "SW Day 5 Average Distance (m)"
    ]

    for col in learning_cols:
        learning[col] = pd.to_numeric(learning[col], errors="coerce")

    # -----------------------------
    # 5. COMPUTE LEARNING SLOPE
    # -----------------------------
    learning_days = np.array([1, 2, 3, 4, 5])


    def compute_learning_slope(row):
        y = row[learning_cols].values.astype(float)
        return np.polyfit(learning_days, y, 1)[0]


    learning["learning_slope"] = learning.apply(compute_learning_slope, axis=1)

    print("\nLearning slope preview:")
    print(learning[["Animal_ID", "learning_slope"]].head())

    # -----------------------------
    # 6. LOAD PROBE DATA
    # -----------------------------
    probe = pd.read_csv("mwm_probe_data.csv")
    probe.columns = probe.columns.str.strip()

    probe = probe[[
        "mouse_ID",
        "day5_SW_TOTAL",
        "day8_SW_TOTAL"
    ]].copy()

    probe = probe.rename(columns={"mouse_ID": "Animal_ID"})

    probe["day5_SW_TOTAL"] = pd.to_numeric(probe["day5_SW_TOTAL"],
                                           errors="coerce")
    probe["day8_SW_TOTAL"] = pd.to_numeric(probe["day8_SW_TOTAL"],
                                           errors="coerce")

    print("\nProbe preview:")
    print(probe.head())

    # -----------------------------
    # 7. MERGE ALL DATASETS
    # -----------------------------
    merged = weights.merge(learning[["Animal_ID", "learning_slope"]],
                           on="Animal_ID", how="inner")
    merged = merged.merge(probe, on="Animal_ID", how="inner")

    print("\nMerged preview:")
    print(merged.head())

    # save merged data
    merged.to_csv("weight_behavior_merged.csv", index=False)

    # -----------------------------
    # 8. SCATTER PLOT: DELTA WEIGHT VS LEARNING SLOPE
    # -----------------------------
    plt.figure(figsize=(6, 5))
    sns.regplot(data=merged, x="delta_weight", y="learning_slope")
    plt.title("Weight Change vs Learning Slope")
    plt.xlabel("Delta Weight")
    plt.ylabel("Learning Slope")
    plt.tight_layout()
    plt.savefig("figures/delta_weight_vs_learning_slope.png", dpi=300)
    plt.show()

    # -----------------------------
    # 9. SCATTER PLOT: DELTA WEIGHT VS DAY 5 SW%
    # -----------------------------
    plt.figure(figsize=(6, 5))
    sns.regplot(data=merged, x="delta_weight", y="day5_SW_TOTAL")
    plt.title("Weight Change vs Day 5 SW Percentage")
    plt.xlabel("Delta Weight")
    plt.ylabel("Day 5 SW/Total")
    plt.tight_layout()
    plt.savefig("figures/delta_weight_vs_day5_swtotal.png", dpi=300)
    plt.show()

    # -----------------------------
    # 10. SCATTER PLOT: DELTA WEIGHT VS DAY 8 SW%
    # -----------------------------
    plt.figure(figsize=(6, 5))
    sns.regplot(data=merged, x="delta_weight", y="day8_SW_TOTAL")
    plt.title("Weight Change vs Day 8 SW Percentage")
    plt.xlabel("Delta Weight")
    plt.ylabel("Day 8 SW/Total")
    plt.tight_layout()
    plt.savefig("figures/delta_weight_vs_day8_swtotal.png", dpi=300)
    plt.show()

    # -----------------------------
    # 11. CORRELATIONS
    # -----------------------------
    r1, p1 = pearsonr(merged["delta_weight"], merged["learning_slope"])
    r2, p2 = pearsonr(merged["delta_weight"], merged["day5_SW_TOTAL"])
    r3, p3 = pearsonr(merged["delta_weight"], merged["day8_SW_TOTAL"])

    corr_results = pd.DataFrame({
        "comparison": [
            "delta_weight vs learning_slope",
            "delta_weight vs day5_SW_TOTAL",
            "delta_weight vs day8_SW_TOTAL"
        ],
        "r": [r1, r2, r3],
        "p_value": [p1, p2, p3],
        "N": [len(merged), len(merged), len(merged)]
    })

    corr_results.to_csv("weight_behavior_correlations.csv", index=False)

    print("\nCorrelation results:")
    print(corr_results)

    # -----------------------------
    # 12. OPTIONAL LINEAR MODELS
    # -----------------------------
    lm1 = smf.ols("learning_slope ~ delta_weight", data=merged).fit()
    lm2 = smf.ols("day5_SW_TOTAL ~ delta_weight", data=merged).fit()
    lm3 = smf.ols("day8_SW_TOTAL ~ delta_weight", data=merged).fit()

    glm_results = []

    for name, model in [
        ("learning_slope ~ delta_weight", lm1),
        ("day5_SW_TOTAL ~ delta_weight", lm2),
        ("day8_SW_TOTAL ~ delta_weight", lm3)
    ]:
        temp = pd.DataFrame({
            "term": model.params.index,
            "estimate": model.params.values,
            "std_error": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values
        })
        temp["model"] = name
        temp["N"] = len(merged)
        glm_results.append(temp)

    glm_results_df = pd.concat(glm_results, ignore_index=True)
    glm_results_df.to_csv("weight_behavior_glm_results.csv", index=False)

    print("\nSaved files:")
    print("- weight_behavior_merged.csv")
    print("- weight_behavior_correlations.csv")
    print("- weight_behavior_glm_results.csv")
    print("- figures/delta_weight_vs_learning_slope.png")
    print("- figures/delta_weight_vs_day5_swtotal.png")
    print("- figures/delta_weight_vs_day8_swtotal.png")
