
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = "mwm_probe_data.csv"
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

DAY_COLORS = {"Day 5": "#55B748", "Day 8": "#4C97D7"}
GENO_LABELS = {
    "APOE22HN": "APOE ε2",
    "APOE33HN": "APOE ε3",
    "APOE44HN": "APOE ε4",
}
GENO_ORDER = ["APOE ε2", "APOE ε3", "APOE ε4"]

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
    df["genotype_label"] = df["genotype"].map(GENO_LABELS)
    return df

def mean_se(x):
    x = np.asarray(x, dtype=float)
    return np.mean(x), np.std(x, ddof=1) / np.sqrt(len(x))

def long_total(df):
    out = df.melt(
        id_vars=["mouse_ID", "genotype", "genotype_label", "Age", "sex"],
        value_vars=["day5_Totaldistance", "day8_Totaldistance"],
        var_name="day",
        value_name="Total_Distance",
    )
    out["Day"] = out["day"].map({
        "day5_Totaldistance": "Day 5",
        "day8_Totaldistance": "Day 8",
    })
    return out

def long_sw(df):
    out = df.melt(
        id_vars=["mouse_ID", "genotype", "genotype_label", "Age", "sex"],
        value_vars=["day5_SW_TOTAL", "day8_SW_TOTAL"],
        var_name="day",
        value_name="SW_Total",
    )
    out["Day"] = out["day"].map({
        "day5_SW_TOTAL": "Day 5",
        "day8_SW_TOTAL": "Day 8",
    })
    return out

def violin_overall_total(df_long):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sns.violinplot(
        data=df_long, x="Day", y="Total_Distance", hue="Day",
        palette=DAY_COLORS, inner=None, linewidth=1.8, cut=0, legend=False, ax=ax
    )
    sns.boxplot(
        data=df_long, x="Day", y="Total_Distance", width=0.15,
        showcaps=True, boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.8},
        whiskerprops={"linewidth": 1.8}, medianprops={"color": "black", "linewidth": 2},
        showfliers=False, ax=ax
    )
    sns.stripplot(
        data=df_long, x="Day", y="Total_Distance", color="black",
        alpha=0.7, size=7, jitter=0.08, ax=ax
    )
    ax.set_title("Probe: Total Distance (Day 5 vs Day 8)")
    ax.set_xlabel("")
    ax.set_ylabel("Total Distance")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "total_distance_day5_vs_day8_violin.png", dpi=300)
    plt.close(fig)

def violin_total_by_genotype(df_long):
    g = sns.catplot(
        data=df_long, x="Day", y="Total_Distance", hue="Day", col="genotype_label",
        kind="violin", inner=None, palette=DAY_COLORS, linewidth=1.8, cut=0,
        height=5.2, aspect=0.8, legend=False, sharey=True, col_order=GENO_ORDER
    )
    for ax, geno in zip(g.axes.flat, GENO_ORDER):
        sub = df_long[df_long["genotype_label"] == geno]
        sns.boxplot(
            data=sub, x="Day", y="Total_Distance", width=0.15,
            showcaps=True, boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 1.5}, medianprops={"color": "black", "linewidth": 2},
            showfliers=False, ax=ax
        )
        sns.stripplot(
            data=sub, x="Day", y="Total_Distance", color="black",
            alpha=0.7, size=6, jitter=0.08, ax=ax
        )
        ax.set_xlabel("")
        ax.set_title(geno, fontweight="bold")
    g.set_ylabels("Total Distance")
    g.figure.suptitle("Probe: Total Distance (Day 5 vs Day 8) by Genotype", y=1.02, fontsize=18)
    g.figure.tight_layout()
    g.figure.savefig(OUTDIR / "total_distance_day5_vs_day8_by_genotype_violin.png", dpi=300, bbox_inches="tight")
    plt.close(g.figure)

def violin_overall_sw(df_long):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    sns.violinplot(
        data=df_long, x="Day", y="SW_Total", hue="Day",
        palette=DAY_COLORS, inner=None, linewidth=1.8, cut=0, legend=False, ax=ax
    )
    sns.boxplot(
        data=df_long, x="Day", y="SW_Total", width=0.15,
        showcaps=True, boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.8},
        whiskerprops={"linewidth": 1.8}, medianprops={"color": "black", "linewidth": 2},
        showfliers=False, ax=ax
    )
    sns.stripplot(
        data=df_long, x="Day", y="SW_Total", color="black",
        alpha=0.7, size=7, jitter=0.08, ax=ax
    )
    ax.set_title("Probe: SW/Total Distance (Day 5 vs Day 8)")
    ax.set_xlabel("")
    ax.set_ylabel("SW Distance / Total Distance")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "sw_total_day5_vs_day8_violin.png", dpi=300)
    plt.close(fig)

def violin_sw_by_genotype(df_long):
    g = sns.catplot(
        data=df_long, x="Day", y="SW_Total", hue="Day", col="genotype_label",
        kind="violin", inner=None, palette=DAY_COLORS, linewidth=1.8, cut=0,
        height=5.2, aspect=0.8, legend=False, sharey=True, col_order=GENO_ORDER
    )
    for ax, geno in zip(g.axes.flat, GENO_ORDER):
        sub = df_long[df_long["genotype_label"] == geno]
        sns.boxplot(
            data=sub, x="Day", y="SW_Total", width=0.15,
            showcaps=True, boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 1.5}, medianprops={"color": "black", "linewidth": 2},
            showfliers=False, ax=ax
        )
        sns.stripplot(
            data=sub, x="Day", y="SW_Total", color="black",
            alpha=0.7, size=6, jitter=0.08, ax=ax
        )
        ax.set_xlabel("")
        ax.set_title(geno, fontweight="bold")
    g.set_ylabels("SW Distance / Total Distance")
    g.figure.suptitle("Probe: SW/Total Distance (Day 5 vs Day 8) by Genotype", y=1.02, fontsize=18)
    g.figure.tight_layout()
    g.figure.savefig(OUTDIR / "sw_total_day5_vs_day8_by_genotype_violin.png", dpi=300, bbox_inches="tight")
    plt.close(g.figure)

def total_distance_by_genotype_points(df_long):
    fig, ax = plt.subplots(figsize=(10.5, 7))
    x_positions = {g:i for i,g in enumerate(GENO_ORDER)}
    offsets = {"Day 5": -0.14, "Day 8": 0.14}
    for day in ["Day 5", "Day 8"]:
        for geno in GENO_ORDER:
            sub = df_long[(df_long["Day"] == day) & (df_long["genotype_label"] == geno)]
            x = x_positions[geno] + offsets[day] + np.random.uniform(-0.05, 0.05, len(sub))
            ax.scatter(x, sub["Total_Distance"], s=70, alpha=0.7, color=DAY_COLORS[day])
            m, se = mean_se(sub["Total_Distance"])
            cx = x_positions[geno] + offsets[day]
            ax.errorbar(cx, m, yerr=se, fmt='o', color='black', capsize=6, markersize=14, elinewidth=2.5, capthick=2.5)
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(GENO_ORDER)
    ax.set_title("Total Distance by Genotype")
    ax.set_xlabel("Genotype")
    ax.set_ylabel("Total Distance")
    handles = [
        plt.Line2D([0], [0], marker='o', color='black', label='Day 5',
                   markerfacecolor='black', markersize=10, linestyle=''),
        plt.Line2D([0], [0], marker='o', color='black', label='Day 8',
                   markerfacecolor='black', markersize=10, linestyle=''),
    ]
    ax.legend(handles=handles, title="Probe Day", frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUTDIR / "total_distance_by_genotype_points_means.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    df = load_data()
    td = long_total(df)
    sw = long_sw(df)

    violin_overall_total(td)
    violin_total_by_genotype(td)
    violin_overall_sw(sw)
    violin_sw_by_genotype(sw)
    total_distance_by_genotype_points(td)

    print(f"Saved figures to: {OUTDIR.resolve()}")
