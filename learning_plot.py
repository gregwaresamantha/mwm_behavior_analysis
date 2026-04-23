"""
Created on 3/15/26
@author: samgregware
"""

if __name__ == '__main__':
    from pathlib import Path
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as smf

    # -----------------------------
    # SETTINGS
    # -----------------------------
    DATA_FILE = "mwm_data.csv"
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

    GENO_REF = "APOE33HN"
    SEX_REF = "F"

    sns.set_style("white")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.title_fontsize": 14,
        "legend.fontsize": 12,
    })

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    df["Animal Code"] = df["Animal Code"].astype(str).str.strip()
    df["Genotype"] = df["Genotype"].astype(str).str.strip()
    df["Sex"] = df["Sex"].astype(str).str.strip()
    df["genotype_label"] = df["Genotype"].map(GENO_LABELS)

    # -----------------------------
    # BUILD LEARNING DATASET
    # -----------------------------
    learning_df = df[[
        "Animal Code", "Genotype", "genotype_label", "Sex", "Age (months)",
        "Day1_SW", "Day2_SW", "Day3_SW", "Day4_SW", "Day5_SW",
        "Day1_Totaldistance", "Day2_Totaldistance", "Day3_Totaldistance",
        "Day4_Totaldistance", "Day5_Totaldistance"
    ]].copy()

    numeric_cols = [
        "Day1_SW", "Day2_SW", "Day3_SW", "Day4_SW", "Day5_SW",
        "Day1_Totaldistance", "Day2_Totaldistance", "Day3_Totaldistance",
        "Day4_Totaldistance", "Day5_Totaldistance"
    ]
    for col in numeric_cols:
        learning_df[col] = pd.to_numeric(learning_df[col], errors="coerce")

    # compute SW / TOTAL for each day
    for d in range(1, 6):
        sw_col = f"Day{d}_SW"
        total_col = f"Day{d}_Totaldistance"
        ratio_col = f"Day{d}_ratio"
        learning_df[ratio_col] = learning_df[sw_col] / learning_df[total_col]

    # -----------------------------
    # LONG FORMAT: SW/TOTAL
    # -----------------------------
    ratio_long = learning_df.melt(
        id_vars=["Animal Code", "Genotype", "genotype_label", "Sex", "Age (months)"],
        value_vars=["Day1_ratio", "Day2_ratio", "Day3_ratio", "Day4_ratio", "Day5_ratio"],
        var_name="Day",
        value_name="SW_ratio"
    )

    ratio_long["Day_num"] = ratio_long["Day"].str.extract(r"(\d+)").astype(int)
    ratio_long["Day_c"] = ratio_long["Day_num"] - ratio_long["Day_num"].mean()
    ratio_long["SW_ratio"] = pd.to_numeric(ratio_long["SW_ratio"], errors="coerce")
    ratio_long = ratio_long.dropna(subset=["SW_ratio", "genotype_label"]).copy()
    ratio_long["MouseID"] = ratio_long["Animal Code"].astype("category")

    # -----------------------------
    # LONG FORMAT: TOTAL DISTANCE
    # -----------------------------
    total_long = learning_df.melt(
        id_vars=["Animal Code", "Genotype", "genotype_label", "Sex", "Age (months)"],
        value_vars=[
            "Day1_Totaldistance",
            "Day2_Totaldistance",
            "Day3_Totaldistance",
            "Day4_Totaldistance",
            "Day5_Totaldistance"
        ],
        var_name="Day",
        value_name="Total_distance"
    )

    total_long["Day_num"] = total_long["Day"].str.extract(r"(\d+)").astype(int)
    total_long["Day_c"] = total_long["Day_num"] - total_long["Day_num"].mean()
    total_long["Total_distance"] = pd.to_numeric(total_long["Total_distance"], errors="coerce")
    total_long = total_long.dropna(subset=["Total_distance", "genotype_label"]).copy()
    total_long["MouseID"] = total_long["Animal Code"].astype("category")

    # -----------------------------
    # MIXED MODEL HELPER
    # -----------------------------
    def run_mixed_model(data, outcome_name, outfile):
        formula = (
            f"{outcome_name} ~ "
            f"Day_c * C(Genotype, Treatment(reference='{GENO_REF}')) + "
            f"Day_c * C(Sex, Treatment(reference='{SEX_REF}'))"
        )

        model = smf.mixedlm(
            formula,
            data=data,
            groups=data["MouseID"]
        )
        result = model.fit(reml=False)

        results_df = pd.DataFrame({
            "term": result.params.index,
            "estimate": result.params.values,
            "std_error": result.bse.values,
            "z_value": result.tvalues.values,
            "p_value": result.pvalues.values
        })
        results_df["model"] = f"MixedLM: {formula} + (1|MouseID)"
        results_df["N_mice"] = data["MouseID"].nunique()
        results_df["N_obs"] = data.shape[0]
        results_df.to_csv(RESULTS_DIR / outfile, index=False)

        return result, results_df

    # -----------------------------
    # RUN MIXED MODELS
    # -----------------------------
    ratio_model, ratio_results = run_mixed_model(
        ratio_long,
        "SW_ratio",
        "learning_sw_ratio_mixedlm_results.csv"
    )

    total_model, total_results = run_mixed_model(
        total_long,
        "Total_distance",
        "learning_total_distance_mixedlm_results.csv"
    )

    # -----------------------------
    # EXTRACT KEY P-VALUES FOR ANNOTATION
    # -----------------------------
    def extract_pvals(results_df):
        day_p = results_df.loc[results_df["term"] == "Day_c", "p_value"]
        geno22_day_p = results_df.loc[
            results_df["term"] == f"Day_c:C(Genotype, Treatment(reference='{GENO_REF}'))[T.APOE22HN]",
            "p_value"
        ]
        geno44_day_p = results_df.loc[
            results_df["term"] == f"Day_c:C(Genotype, Treatment(reference='{GENO_REF}'))[T.APOE44HN]",
            "p_value"
        ]

        def safe_get(series):
            return float(series.iloc[0]) if len(series) else float("nan")

        return {
            "day_p": safe_get(day_p),
            "geno22_day_p": safe_get(geno22_day_p),
            "geno44_day_p": safe_get(geno44_day_p),
        }

    ratio_pvals = extract_pvals(ratio_results)
    total_pvals = extract_pvals(total_results)

    # -----------------------------
    # FIGURE 1: SW / TOTAL LEARNING CURVE
    # -----------------------------
    plt.figure(figsize=(8.5, 6))

    sns.lineplot(
        data=ratio_long,
        x="Day_num",
        y="SW_ratio",
        hue="genotype_label",
        hue_order=GENO_ORDER,
        estimator="mean",
        errorbar="se",
        marker="o",
        linewidth=2.5
    )

    plt.title("Learning Curve: SW / Total Distance")
    plt.xlabel("Training Day")
    plt.ylabel("SW Distance / Total Distance")
    plt.xticks([1, 2, 3, 4, 5])

    ratio_text = (
        f"Mixed model\n"
        f"Day p = {ratio_pvals['day_p']:.3g}\n"
        f"Day×ε2 vs ε3 p = {ratio_pvals['geno22_day_p']:.3g}\n"
        f"Day×ε4 vs ε3 p = {ratio_pvals['geno44_day_p']:.3g}"
    )

    plt.text(
        0.02, 0.98, ratio_text,
        transform=plt.gca().transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    sns.despine()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "learning_curve_sw_ratio.png", dpi=300)
    plt.show()

    # -----------------------------
    # FIGURE 2: TOTAL DISTANCE LEARNING CURVE
    # -----------------------------
    plt.figure(figsize=(8.5, 6))

    sns.lineplot(
        data=total_long,
        x="Day_num",
        y="Total_distance",
        hue="genotype_label",
        hue_order=GENO_ORDER,
        estimator="mean",
        errorbar="se",
        marker="o",
        linewidth=2.5
    )

    plt.title("Learning Curve: Total Distance Across Training Days")
    plt.xlabel("Training Day")
    plt.ylabel("Total Distance")
    plt.xticks([1, 2, 3, 4, 5])

    total_text = (
        f"Mixed model\n"
        f"Day p = {total_pvals['day_p']:.3g}\n"
        f"Day×ε2 vs ε3 p = {total_pvals['geno22_day_p']:.3g}\n"
        f"Day×ε4 vs ε3 p = {total_pvals['geno44_day_p']:.3g}"
    )

    plt.text(
        0.02, 0.98, total_text,
        transform=plt.gca().transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    sns.despine()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "learning_curve_total_distance.png", dpi=300)
    plt.show()

    print("\nSaved:")
    print("- figures/learning_curve_sw_ratio.png")
    print("- figures/learning_curve_total_distance.png")
    print("- results/learning_sw_ratio_mixedlm_results.csv")
    print("- results/learning_total_distance_mixedlm_results.csv")