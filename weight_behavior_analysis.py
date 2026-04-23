"""
Created on 3/15/26
@author: samgregware
"""

if __name__ == '__main__':
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
    FIG_DIR = Path("figures")
    FIG_DIR.mkdir(exist_ok=True)

    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    GENO_REF = "APOE33HN"
    SEX_REF = "F"

    DAY_COLORS = {"Day 5": "#55B748", "Day 8": "#4C97D7"}
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
    # 1. LOAD WEIGHTS
    # -----------------------------
    weights = pd.read_csv("mouse_body_weights_sema.csv", sep=",")
    weights.columns = weights.columns.str.strip()

    weights = weights[[
        "Animal_ID",
        "AnimalWeight_g_100825",
        "AnimalWeight_g_121025"
    ]].copy()

    weights["Animal_ID"] = weights["Animal_ID"].astype(str).str.strip()
    weights["AnimalWeight_g_100825"] = pd.to_numeric(weights["AnimalWeight_g_100825"], errors="coerce")
    weights["AnimalWeight_g_121025"] = pd.to_numeric(weights["AnimalWeight_g_121025"], errors="coerce")
    weights = weights.dropna(subset=["Animal_ID", "AnimalWeight_g_100825", "AnimalWeight_g_121025"]).copy()

    # percent body weight change
    weights["delta_mass"] = (
        (weights["AnimalWeight_g_121025"] - weights["AnimalWeight_g_100825"])
        / weights["AnimalWeight_g_100825"]
    ) * 100

    # -----------------------------
    # 2. LOAD MWM DATA
    # -----------------------------
    mwm = pd.read_csv("mwm_data.csv")
    mwm.columns = mwm.columns.str.strip()

    mwm["Animal Code"] = mwm["Animal Code"].astype(str).str.strip()
    mwm["Genotype"] = mwm["Genotype"].astype(str).str.strip()
    mwm["Sex"] = mwm["Sex"].astype(str).str.strip()
    mwm["genotype_label"] = mwm["Genotype"].map(GENO_LABELS)

    # probe columns
    probe = mwm[[
        "Animal Code",
        "Genotype",
        "Sex",
        "genotype_label",
        "Distance Probe Day 5",
        "Distance Probe Day 8",
        "SW Distance Probe Day 5",
        "SW Distance Probe Day 8"
    ]].copy()

    probe = probe.rename(columns={"Animal Code": "Animal_ID"})
    probe["Animal_ID"] = probe["Animal_ID"].astype(str).str.strip()

    numeric_probe_cols = [
        "Distance Probe Day 5",
        "Distance Probe Day 8",
        "SW Distance Probe Day 5",
        "SW Distance Probe Day 8"
    ]
    for col in numeric_probe_cols:
        probe[col] = pd.to_numeric(probe[col], errors="coerce")

    probe["day5_SW_TOTAL"] = probe["SW Distance Probe Day 5"] / probe["Distance Probe Day 5"]
    probe["day8_SW_TOTAL"] = probe["SW Distance Probe Day 8"] / probe["Distance Probe Day 8"]
    probe["delta_memory"] = probe["day8_SW_TOTAL"] - probe["day5_SW_TOTAL"]

    probe = probe[[
        "Animal_ID",
        "Genotype",
        "Sex",
        "genotype_label",
        "day5_SW_TOTAL",
        "day8_SW_TOTAL",
        "delta_memory"
    ]].copy()

    # -----------------------------
    # 3. MERGE
    # -----------------------------
    merged = weights.merge(
        probe,
        on="Animal_ID",
        how="inner"
    )

    merged.to_csv(RESULTS_DIR / "probe_memory_merged.csv", index=False)

    print("Merged columns:")
    print(merged.columns.tolist())
    print(f"N merged mice = {len(merged)}")

    # -----------------------------
    # 4. HELPER FUNCTIONS
    # -----------------------------
    def fit_glm(formula, data, label):
        model = smf.ols(formula, data=data).fit()
        beta = model.params.iloc[1] if len(model.params) > 1 else np.nan
        pval = model.pvalues.iloc[1] if len(model.pvalues) > 1 else np.nan
        r2 = model.rsquared
        n = int(model.nobs)

        out = pd.DataFrame({
            "term": model.params.index,
            "estimate": model.params.values,
            "std_error": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values
        })
        out["model"] = label
        out["formula"] = formula
        out["N"] = n
        out["r_squared"] = r2
        return model, out, beta, pval, r2, n

    def make_reg_plot(data, x, y, title, y_label, filename, beta, pval, r2):
        plt.figure(figsize=(6, 5))
        sns.regplot(data=data, x=x, y=y, scatter_kws={"s": 55})

        plt.title(title)
        plt.xlabel("Percent Body Weight Change (%)")
        plt.ylabel(y_label)

        stats_text = f"β = {beta:.3f}\np = {pval:.3g}\nR² = {r2:.3f}"
        plt.text(
            0.05, 0.95, stats_text,
            transform=plt.gca().transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )

        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=300)
        plt.close()

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
    # 5. DELTA MASS REGRESSIONS
    # -----------------------------
    glm_results = []
    summary_rows = []

    glm_specs = [
        (
            "Probe Day 5 ~ delta_mass",
            "day5_SW_TOTAL ~ delta_mass",
            "day5_SW_TOTAL",
            "Delta Mass vs Probe Day 5 SW/TOTAL",
            "Probe Day 5 SW/TOTAL",
            "delta_mass_vs_probe5_swtotal.png"
        ),
        (
            "Probe Day 8 ~ delta_mass",
            "day8_SW_TOTAL ~ delta_mass",
            "day8_SW_TOTAL",
            "Delta Mass vs Probe Day 8 SW/TOTAL",
            "Probe Day 8 SW/TOTAL",
            "delta_mass_vs_probe8_swtotal.png"
        ),
        (
            "Day 8 - Day 5 memory ~ delta_mass",
            "delta_memory ~ delta_mass",
            "delta_memory",
            "Delta Mass vs Change in Memory (Day 8 - Day 5)",
            "Δ Memory (Day 8 - Day 5)",
            "delta_mass_vs_delta_memory.png"
        ),
    ]

    for label, formula, ycol, title, ylabel, fname in glm_specs:
        model, out, beta, pval, r2, n = fit_glm(formula, merged, label)
        glm_results.append(out)

        summary_rows.append({
            "comparison": label,
            "beta_delta_mass": beta,
            "p_value_delta_mass": pval,
            "r_squared": r2,
            "N": n
        })

        make_reg_plot(
            merged, "delta_mass", ycol, title, ylabel, fname,
            beta, pval, r2
        )

    glm_results_df = pd.concat(glm_results, ignore_index=True)
    glm_results_df.to_csv(RESULTS_DIR / "probe_memory_delta_mass_glm_results.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "probe_memory_delta_mass_summary.csv", index=False)

    # -----------------------------
    # 6. LONG FORMAT FOR DAY5 vs DAY8
    # -----------------------------
    probe_long = merged.melt(
        id_vars=["Animal_ID", "Genotype", "Sex", "genotype_label", "delta_mass"],
        value_vars=["day5_SW_TOTAL", "day8_SW_TOTAL"],
        var_name="Day",
        value_name="SW_ratio"
    )

    probe_long["Day"] = probe_long["Day"].map({
        "day5_SW_TOTAL": "Day 5",
        "day8_SW_TOTAL": "Day 8"
    })
    probe_long["Day_num"] = probe_long["Day"].map({"Day 5": 5, "Day 8": 8})
    probe_long["Day_c"] = probe_long["Day_num"] - probe_long["Day_num"].mean()

    probe_long["SW_ratio"] = pd.to_numeric(probe_long["SW_ratio"], errors="coerce")
    probe_long = probe_long.dropna(subset=["SW_ratio"]).copy()

    probe_long["Animal_ID"] = probe_long["Animal_ID"].astype("category")
    probe_long["Genotype"] = probe_long["Genotype"].astype("category")
    probe_long["Sex"] = probe_long["Sex"].astype("category")

    # -----------------------------
    # 7. MIXED MODEL: DAY5 vs DAY8 OVERALL
    # -----------------------------
    mixed_formula = (
        f"SW_ratio ~ "
        f"C(Day, Treatment(reference='Day 5')) * C(Genotype, Treatment(reference='{GENO_REF}')) + "
        f"C(Day, Treatment(reference='Day 5')) * C(Sex, Treatment(reference='{SEX_REF}'))"
    )

    mixed_model = smf.mixedlm(
        mixed_formula,
        data=probe_long,
        groups=probe_long["Animal_ID"]
    ).fit(reml=False)

    print("\nDay 5 vs Day 8 mixed model:")
    print(mixed_model.summary())

    mixed_results_df = pd.DataFrame({
        "term": mixed_model.params.index,
        "estimate": mixed_model.params.values,
        "std_error": mixed_model.bse.values,
        "z_value": mixed_model.tvalues.values,
        "p_value": mixed_model.pvalues.values
    })
    mixed_results_df["model"] = f"MixedLM: {mixed_formula} + (1|Animal_ID)"
    mixed_results_df["N_mice"] = probe_long["Animal_ID"].nunique()
    mixed_results_df["N_obs"] = probe_long.shape[0]
    mixed_results_df.to_csv(RESULTS_DIR / "probe_day5_day8_mixedlm_results.csv", index=False)

    day_p = mixed_model.pvalues.get(
        "C(Day, Treatment(reference='Day 5'))[T.Day 8]",
        np.nan
    )
    int_22_p = mixed_model.pvalues.get(
        f"C(Day, Treatment(reference='Day 5'))[T.Day 8]:C(Genotype, Treatment(reference='{GENO_REF}'))[T.APOE22HN]",
        np.nan
    )
    int_44_p = mixed_model.pvalues.get(
        f"C(Day, Treatment(reference='Day 5'))[T.Day 8]:C(Genotype, Treatment(reference='{GENO_REF}'))[T.APOE44HN]",
        np.nan
    )

    # -----------------------------
    # 8. DAY 5 and DAY 8 GLMs BY GENOTYPE
    # -----------------------------
    day5_glm = smf.ols(
        f"day5_SW_TOTAL ~ C(Genotype, Treatment(reference='{GENO_REF}')) + C(Sex, Treatment(reference='{SEX_REF}'))",
        data=merged
    ).fit()

    day8_glm = smf.ols(
        f"day8_SW_TOTAL ~ C(Genotype, Treatment(reference='{GENO_REF}')) + C(Sex, Treatment(reference='{SEX_REF}'))",
        data=merged
    ).fit()

    day5_glm_df = pd.DataFrame({
        "term": day5_glm.params.index,
        "estimate": day5_glm.params.values,
        "std_error": day5_glm.bse.values,
        "t_value": day5_glm.tvalues.values,
        "p_value": day5_glm.pvalues.values
    })
    day5_glm_df["model"] = "GLM: day5_SW_TOTAL ~ Genotype + Sex"
    day5_glm_df["N"] = int(day5_glm.nobs)
    day5_glm_df.to_csv(RESULTS_DIR / "probe_day5_glm_results.csv", index=False)

    day8_glm_df = pd.DataFrame({
        "term": day8_glm.params.index,
        "estimate": day8_glm.params.values,
        "std_error": day8_glm.bse.values,
        "t_value": day8_glm.tvalues.values,
        "p_value": day8_glm.pvalues.values
    })
    day8_glm_df["model"] = "GLM: day8_SW_TOTAL ~ Genotype + Sex"
    day8_glm_df["N"] = int(day8_glm.nobs)
    day8_glm_df.to_csv(RESULTS_DIR / "probe_day8_glm_results.csv", index=False)

    # -----------------------------
    # 9. BOXPLOT STATS CSV
    # -----------------------------
    box_stats = compute_box_stats(probe_long, "SW_ratio", "SW/Total Probe")
    box_stats.to_csv(RESULTS_DIR / "probe_day5_day8_boxplot_stats.csv", index=False)

    # -----------------------------
    # 10. OVERALL DAY5 vs DAY8 PLOT
    # -----------------------------
    plt.figure(figsize=(7, 6))
    ax = sns.violinplot(
        data=probe_long,
        x="Day",
        y="SW_ratio",
        hue="Day",
        palette=DAY_COLORS,
        inner=None,
        cut=0,
        linewidth=1.6,
        legend=False
    )
    sns.boxplot(
        data=probe_long,
        x="Day",
        y="SW_ratio",
        width=0.15,
        showcaps=True,
        boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.6},
        whiskerprops={"linewidth": 1.6},
        medianprops={"color": "black", "linewidth": 2},
        showfliers=False,
        ax=ax
    )
    sns.stripplot(
        data=probe_long,
        x="Day",
        y="SW_ratio",
        color="black",
        alpha=0.7,
        jitter=0.08,
        ax=ax
    )

    stats_text = (
        f"Mixed model\n"
        f"Day 8 vs Day 5 p = {day_p:.3g}\n"
        f"Day×ε2 vs ε3 p = {int_22_p:.3g}\n"
        f"Day×ε4 vs ε3 p = {int_44_p:.3g}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    overall_day5 = probe_long.loc[probe_long["Day"] == "Day 5", "SW_ratio"].dropna()
    overall_day8 = probe_long.loc[probe_long["Day"] == "Day 8", "SW_ratio"].dropna()
    qtext = (
        f"Day 5: Q1={overall_day5.quantile(0.25):.3f}, "
        f"Med={overall_day5.median():.3f}, Q3={overall_day5.quantile(0.75):.3f}\n"
        f"Day 8: Q1={overall_day8.quantile(0.25):.3f}, "
        f"Med={overall_day8.median():.3f}, Q3={overall_day8.quantile(0.75):.3f}"
    )
    ax.text(
        0.98, 0.98, qtext,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    ax.set_title("Probe Memory: Day 5 vs Day 8")
    ax.set_xlabel("")
    ax.set_ylabel("SW Distance / Total Distance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "probe_day5_vs_day8_overall.png", dpi=300)
    plt.close()

    # -----------------------------
    # 11. DAY5 vs DAY8 BY GENOTYPE
    # with within-genotype paired test
    # -----------------------------
    g = sns.catplot(
        data=probe_long,
        x="Day",
        y="SW_ratio",
        hue="Day",
        col="genotype_label",
        col_order=GENO_ORDER,
        kind="violin",
        inner=None,
        palette=DAY_COLORS,
        cut=0,
        linewidth=1.6,
        height=5.2,
        aspect=0.85,
        legend=False,
        sharey=True
    )

    for ax, geno in zip(g.axes.flat, GENO_ORDER):
        sub = probe_long[probe_long["genotype_label"] == geno]

        sns.boxplot(
            data=sub,
            x="Day",
            y="SW_ratio",
            width=0.15,
            showcaps=True,
            boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 1.5},
            medianprops={"color": "black", "linewidth": 2},
            showfliers=False,
            ax=ax
        )
        sns.stripplot(
            data=sub,
            x="Day",
            y="SW_ratio",
            color="black",
            alpha=0.7,
            jitter=0.08,
            ax=ax
        )

        day5_vals = (
            sub[sub["Day"] == "Day 5"][["Animal_ID", "SW_ratio"]]
            .rename(columns={"SW_ratio": "d5"})
        )
        day8_vals = (
            sub[sub["Day"] == "Day 8"][["Animal_ID", "SW_ratio"]]
            .rename(columns={"SW_ratio": "d8"})
        )
        paired = day5_vals.merge(day8_vals, on="Animal_ID", how="inner")

        paired_p = np.nan
        if len(paired) >= 2:
            _, paired_p = ttest_rel(paired["d5"], paired["d8"])

        q1_5 = paired["d5"].quantile(0.25) if len(paired) else np.nan
        med_5 = paired["d5"].median() if len(paired) else np.nan
        q3_5 = paired["d5"].quantile(0.75) if len(paired) else np.nan

        q1_8 = paired["d8"].quantile(0.25) if len(paired) else np.nan
        med_8 = paired["d8"].median() if len(paired) else np.nan
        q3_8 = paired["d8"].quantile(0.75) if len(paired) else np.nan

        text = (
            f"Paired Day5 vs Day8 p = {paired_p:.3g}\n"
            f"Day5: {q1_5:.2f} | {med_5:.2f} | {q3_5:.2f}\n"
            f"Day8: {q1_8:.2f} | {med_8:.2f} | {q3_8:.2f}"
        )
        ax.text(
            0.02, 0.98, text,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )

        ax.set_title(geno, fontweight="bold")
        ax.set_xlabel("")

    overall_text = (
        f"Overall mixed model\n"
        f"Day 8 vs Day 5 p = {day_p:.3g}\n"
        f"Day×ε2 vs ε3 p = {int_22_p:.3g}\n"
        f"Day×ε4 vs ε3 p = {int_44_p:.3g}"
    )
    g.axes.flat[0].text(
        0.02, 0.02, overall_text,
        transform=g.axes.flat[0].transAxes,
        ha="left", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    g.set_ylabels("SW Distance / Total Distance")
    g.figure.suptitle("Probe Memory: Day 5 vs Day 8 by Genotype", y=1.02, fontsize=18)
    g.figure.tight_layout()
    g.figure.savefig(FIG_DIR / "probe_day5_vs_day8_by_genotype.png", dpi=300, bbox_inches="tight")
    plt.close(g.figure)

    print("\nSaved files:")
    print("- results/probe_memory_merged.csv")
    print("- results/probe_memory_delta_mass_glm_results.csv")
    print("- results/probe_memory_delta_mass_summary.csv")
    print("- results/probe_day5_day8_mixedlm_results.csv")
    print("- results/probe_day5_glm_results.csv")
    print("- results/probe_day8_glm_results.csv")
    print("- results/probe_day5_day8_boxplot_stats.csv")
    print("- figures/delta_mass_vs_probe5_swtotal.png")
    print("- figures/delta_mass_vs_probe8_swtotal.png")
    print("- figures/delta_mass_vs_delta_memory.png")
    print("- figures/probe_day5_vs_day8_overall.png")
    print("- figures/probe_day5_vs_day8_by_genotype.png")