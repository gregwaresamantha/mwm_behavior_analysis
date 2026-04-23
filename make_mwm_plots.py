from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

DATA_FILE = "mwm_data.csv"
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DAY_COLORS = {"Day 5": "#55B748", "Day 8": "#4C97D7"}
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
    "legend.title_fontsize": 16,
    "legend.fontsize": 13,
})

def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    df["Animal Code"] = df["Animal Code"].astype(str).str.strip()
    df["Genotype"] = df["Genotype"].astype(str).str.strip()
    df["Sex"] = df["Sex"].astype(str).str.strip()
    df["genotype_label"] = df["Genotype"].map(GENO_LABELS)

    probe_numeric_cols = [
        "Distance Probe Day 5",
        "Distance Probe Day 8",
        "SW Distance Probe Day 5",
        "SW Distance Probe Day 8",
    ]
    for col in probe_numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["day5_SW_TOTAL"] = df["SW Distance Probe Day 5"] / df["Distance Probe Day 5"]
    df["day8_SW_TOTAL"] = df["SW Distance Probe Day 8"] / df["Distance Probe Day 8"]

    return df

def mean_se(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return np.mean(x), np.std(x, ddof=1) / np.sqrt(len(x))

def long_total(df):
    out = df.melt(
        id_vars=["Animal Code", "Genotype", "genotype_label", "Age (months)", "Sex"],
        value_vars=["Distance Probe Day 5", "Distance Probe Day 8"],
        var_name="day",
        value_name="Total_Distance",
    )
    out["Day"] = out["day"].map({
        "Distance Probe Day 5": "Day 5",
        "Distance Probe Day 8": "Day 8",
    })
    out["Day_num"] = out["Day"].map({"Day 5": 5, "Day 8": 8})
    out["Day_c"] = out["Day_num"] - out["Day_num"].mean()
    out["Total_Distance"] = pd.to_numeric(out["Total_Distance"], errors="coerce")
    out = out.dropna(subset=["Total_Distance"]).copy()
    out["MouseID"] = out["Animal Code"].astype("category")
    return out

def long_sw(df):
    out = df.melt(
        id_vars=["Animal Code", "Genotype", "genotype_label", "Age (months)", "Sex"],
        value_vars=["day5_SW_TOTAL", "day8_SW_TOTAL"],
        var_name="day",
        value_name="SW_Total",
    )
    out["Day"] = out["day"].map({
        "day5_SW_TOTAL": "Day 5",
        "day8_SW_TOTAL": "Day 8",
    })
    out["Day_num"] = out["Day"].map({"Day 5": 5, "Day 8": 8})
    out["Day_c"] = out["Day_num"] - out["Day_num"].mean()
    out["SW_Total"] = pd.to_numeric(out["SW_Total"], errors="coerce")
    out = out.dropna(subset=["SW_Total"]).copy()
    out["MouseID"] = out["Animal Code"].astype("category")
    return out

def export_mixedlm(result, label, outfile, n_mice, n_obs):
    out = pd.DataFrame({
        "term": result.params.index,
        "estimate": result.params.values,
        "std_error": result.bse.values,
        "z_value": result.tvalues.values,
        "p_value": result.pvalues.values
    })
    out["model"] = label
    out["N_mice"] = n_mice
    out["N_obs"] = n_obs
    out.to_csv(RESULTS_DIR / outfile, index=False)

def run_probe_mixed_model(df_long, outcome_name, outfile):
    formula = (
        f"{outcome_name} ~ "
        f"Day_c * C(Genotype, Treatment(reference='{GENO_REF}')) + "
        f"Day_c * C(Sex, Treatment(reference='{SEX_REF}'))"
    )

    model = smf.mixedlm(
        formula,
        data=df_long,
        groups=df_long["MouseID"]
    )
    result = model.fit(reml=False)

    export_mixedlm(
        result,
        f"MixedLM: {formula} + (1|MouseID)",
        outfile,
        df_long["MouseID"].nunique(),
        df_long.shape[0]
    )
    return result

def extract_pvals(result):
    p = result.pvalues

    def safe_get(term):
        return float(p[term]) if term in p.index else np.nan

    return {
        "day_p": safe_get("Day_c"),
        "geno22_day_p": safe_get(
            f"Day_c:C(Genotype, Treatment(reference='{GENO_REF}'))[T.APOE22HN]"
        ),
        "geno44_day_p": safe_get(
            f"Day_c:C(Genotype, Treatment(reference='{GENO_REF}'))[T.APOE44HN]"
        ),
    }

def add_stats_text(ax, pvals):
    stats_text = (
        "Mixed model\n"
        f"Day p = {pvals['day_p']:.3g}\n"
        f"Day×ε2 vs ε3 p = {pvals['geno22_day_p']:.3g}\n"
        f"Day×ε4 vs ε3 p = {pvals['geno44_day_p']:.3g}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

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

def add_box_stats_overall(ax, df_long, value_col):
    lines = []
    for day in ["Day 5", "Day 8"]:
        sub = df_long[df_long["Day"] == day][value_col].dropna()
        if len(sub) == 0:
            continue
        q1 = sub.quantile(0.25)
        med = sub.median()
        q3 = sub.quantile(0.75)
        lines.append(f"{day}: Q1={q1:.3f}, Med={med:.3f}, Q3={q3:.3f}")

    text = "Box stats\n" + "\n".join(lines)
    ax.text(
        0.98, 0.98, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

def add_box_stats_facet(ax, df_long, value_col, geno):
    sub = df_long[df_long["genotype_label"] == geno]
    lines = []
    for day in ["Day 5", "Day 8"]:
        vals = sub[sub["Day"] == day][value_col].dropna()
        if len(vals) == 0:
            continue
        q1 = vals.quantile(0.25)
        med = vals.median()
        q3 = vals.quantile(0.75)
        lines.append(f"{day}: {q1:.2f} | {med:.2f} | {q3:.2f}")

    text = "Q1 | Med | Q3\n" + "\n".join(lines)
    ax.text(
        0.98, 0.98, text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.80)
    )

def violin_overall_total(df_long, pvals):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sns.violinplot(
        data=df_long, x="Day", y="Total_Distance", hue="Day",
        palette=DAY_COLORS, inner=None, linewidth=1.8, cut=0, legend=False, ax=ax
    )
    sns.boxplot(
        data=df_long, x="Day", y="Total_Distance", width=0.15,
        showcaps=True,
        boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.8},
        whiskerprops={"linewidth": 1.8},
        medianprops={"color": "black", "linewidth": 2},
        showfliers=False, ax=ax
    )
    sns.stripplot(
        data=df_long, x="Day", y="Total_Distance", color="black",
        alpha=0.7, size=7, jitter=0.08, ax=ax
    )
    ax.set_title("Probe: Total Distance (Day 5 vs Day 8)")
    ax.set_xlabel("")
    ax.set_ylabel("Total Distance")
    add_stats_text(ax, pvals)
    add_box_stats_overall(ax, df_long, "Total_Distance")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "total_distance_day5_vs_day8_violin.png", dpi=300)
    plt.close(fig)

def violin_total_by_genotype(df_long, pvals):
    g = sns.catplot(
        data=df_long, x="Day", y="Total_Distance", hue="Day", col="genotype_label",
        kind="violin", inner=None, palette=DAY_COLORS, linewidth=1.8, cut=0,
        height=5.2, aspect=0.8, legend=False, sharey=True, col_order=GENO_ORDER
    )
    for ax, geno in zip(g.axes.flat, GENO_ORDER):
        sub = df_long[df_long["genotype_label"] == geno]
        sns.boxplot(
            data=sub, x="Day", y="Total_Distance", width=0.15,
            showcaps=True,
            boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 1.5},
            medianprops={"color": "black", "linewidth": 2},
            showfliers=False, ax=ax
        )
        sns.stripplot(
            data=sub, x="Day", y="Total_Distance", color="black",
            alpha=0.7, size=6, jitter=0.08, ax=ax
        )
        ax.set_xlabel("")
        ax.set_title(geno, fontweight="bold")
        add_box_stats_facet(ax, df_long, "Total_Distance", geno)

    stats_text = (
        "Mixed model\n"
        f"Day p = {pvals['day_p']:.3g}\n"
        f"Day×ε2 vs ε3 p = {pvals['geno22_day_p']:.3g}\n"
        f"Day×ε4 vs ε3 p = {pvals['geno44_day_p']:.3g}"
    )
    g.axes.flat[0].text(
        0.02, 0.98, stats_text,
        transform=g.axes.flat[0].transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    g.set_ylabels("Total Distance")
    g.figure.suptitle("Probe: Total Distance (Day 5 vs Day 8) by Genotype", y=1.02, fontsize=18)
    g.figure.tight_layout()
    g.figure.savefig(
        OUTDIR / "total_distance_day5_vs_day8_by_genotype_violin.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(g.figure)

def violin_overall_sw(df_long, pvals):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sns.violinplot(
        data=df_long, x="Day", y="SW_Total", hue="Day",
        palette=DAY_COLORS, inner=None, linewidth=1.8, cut=0, legend=False, ax=ax
    )
    sns.boxplot(
        data=df_long, x="Day", y="SW_Total", width=0.15,
        showcaps=True,
        boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.8},
        whiskerprops={"linewidth": 1.8},
        medianprops={"color": "black", "linewidth": 2},
        showfliers=False, ax=ax
    )
    sns.stripplot(
        data=df_long, x="Day", y="SW_Total", color="black",
        alpha=0.7, size=7, jitter=0.08, ax=ax
    )
    ax.set_title("Probe: SW/Total Distance (Day 5 vs Day 8)")
    ax.set_xlabel("")
    ax.set_ylabel("SW Distance / Total Distance")
    add_stats_text(ax, pvals)
    add_box_stats_overall(ax, df_long, "SW_Total")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "sw_total_day5_vs_day8_violin.png", dpi=300)
    plt.close(fig)

def violin_sw_by_genotype(df_long, pvals):
    g = sns.catplot(
        data=df_long, x="Day", y="SW_Total", hue="Day", col="genotype_label",
        kind="violin", inner=None, palette=DAY_COLORS, linewidth=1.8, cut=0,
        height=5.2, aspect=0.8, legend=False, sharey=True, col_order=GENO_ORDER
    )
    for ax, geno in zip(g.axes.flat, GENO_ORDER):
        sub = df_long[df_long["genotype_label"] == geno]
        sns.boxplot(
            data=sub, x="Day", y="SW_Total", width=0.15,
            showcaps=True,
            boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 1.5},
            medianprops={"color": "black", "linewidth": 2},
            showfliers=False, ax=ax
        )
        sns.stripplot(
            data=sub, x="Day", y="SW_Total", color="black",
            alpha=0.7, size=6, jitter=0.08, ax=ax
        )
        ax.set_xlabel("")
        ax.set_title(geno, fontweight="bold")
        add_box_stats_facet(ax, df_long, "SW_Total", geno)

    stats_text = (
        "Mixed model\n"
        f"Day p = {pvals['day_p']:.3g}\n"
        f"Day×ε2 vs ε3 p = {pvals['geno22_day_p']:.3g}\n"
        f"Day×ε4 vs ε3 p = {pvals['geno44_day_p']:.3g}"
    )
    g.axes.flat[0].text(
        0.02, 0.98, stats_text,
        transform=g.axes.flat[0].transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    g.set_ylabels("SW Distance / Total Distance")
    g.figure.suptitle("Probe: SW/Total Distance (Day 5 vs Day 8) by Genotype", y=1.02, fontsize=18)
    g.figure.tight_layout()
    g.figure.savefig(
        OUTDIR / "sw_total_day5_vs_day8_by_genotype_violin.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(g.figure)

def total_distance_by_genotype_points(df_long, pvals):
    fig, ax = plt.subplots(figsize=(10.5, 7))
    x_positions = {g: i for i, g in enumerate(GENO_ORDER)}
    offsets = {"Day 5": -0.14, "Day 8": 0.14}

    for day in ["Day 5", "Day 8"]:
        for geno in GENO_ORDER:
            sub = df_long[(df_long["Day"] == day) & (df_long["genotype_label"] == geno)]
            x = x_positions[geno] + offsets[day] + np.random.uniform(-0.05, 0.05, len(sub))
            ax.scatter(x, sub["Total_Distance"], s=70, alpha=0.7, color=DAY_COLORS[day])

            if len(sub) > 1:
                m, se = mean_se(sub["Total_Distance"])
                cx = x_positions[geno] + offsets[day]
                ax.errorbar(
                    cx, m, yerr=se, fmt="o", color="black",
                    capsize=6, markersize=14, elinewidth=2.5, capthick=2.5
                )

    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(GENO_ORDER)
    ax.set_title("Total Distance by Genotype")
    ax.set_xlabel("Genotype")
    ax.set_ylabel("Total Distance")
    add_stats_text(ax, pvals)

    handles = [
        plt.Line2D([0], [0], marker="o", color="black", label="Day 5",
                   markerfacecolor="black", markersize=10, linestyle=""),
        plt.Line2D([0], [0], marker="o", color="black", label="Day 8",
                   markerfacecolor="black", markersize=10, linestyle=""),
    ]
    ax.legend(handles=handles, title="Probe Day", frameon=False,
              loc="center left", bbox_to_anchor=(1.02, 0.5))
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "total_distance_by_genotype_points_means.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    df = load_data()
    td = long_total(df)
    sw = long_sw(df)

    total_result = run_probe_mixed_model(
        td,
        "Total_Distance",
        "probe_total_distance_mixedlm_results.csv"
    )
    sw_result = run_probe_mixed_model(
        sw,
        "SW_Total",
        "probe_sw_total_mixedlm_results.csv"
    )

    total_pvals = extract_pvals(total_result)
    sw_pvals = extract_pvals(sw_result)

    total_stats = compute_box_stats(td, "Total_Distance", "Total Distance")
    sw_stats = compute_box_stats(sw, "SW_Total", "SW/Total")
    box_stats = pd.concat([total_stats, sw_stats], ignore_index=True)
    box_stats.to_csv(RESULTS_DIR / "probe_boxplot_stats.csv", index=False)

    violin_overall_total(td, total_pvals)
    violin_total_by_genotype(td, total_pvals)
    violin_overall_sw(sw, sw_pvals)
    violin_sw_by_genotype(sw, sw_pvals)
    total_distance_by_genotype_points(td, total_pvals)

    print(f"Saved figures to: {OUTDIR.resolve()}")
    print("Saved results to: results/probe_total_distance_mixedlm_results.csv")
    print("Saved results to: results/probe_sw_total_mixedlm_results.csv")
    print("Saved results to: results/probe_boxplot_stats.csv")