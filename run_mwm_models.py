
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

DATA_FILE = "mwm_probe_data.csv"
OUTDIR = Path("results")
OUTDIR.mkdir(exist_ok=True)

GENO_LABELS = {
    "APOE22HN": "APOE_e2",
    "APOE33HN": "APOE_e3",
    "APOE44HN": "APOE_e4",
}

def write_block(f, title, text):
    f.write(f"\n{'='*80}\n{title}\n{'='*80}\n")
    f.write(text)
    if not text.endswith("\n"):
        f.write("\n")

df = pd.read_csv(DATA_FILE)
df["genotype_label"] = df["genotype"].map(GENO_LABELS)

# Long data for repeated-measures models
td_long = df.melt(
    id_vars=["mouse_ID", "genotype_label", "Age", "sex"],
    value_vars=["day5_Totaldistance", "day8_Totaldistance"],
    var_name="raw_day", value_name="Total_Distance"
)
td_long["Day"] = td_long["raw_day"].map({
    "day5_Totaldistance": 5,
    "day8_Totaldistance": 8,
})

sw_long = df.melt(
    id_vars=["mouse_ID", "genotype_label", "Age", "sex"],
    value_vars=["day5_SW_TOTAL", "day8_SW_TOTAL"],
    var_name="raw_day", value_name="SW_Total"
)
sw_long["Day"] = sw_long["raw_day"].map({
    "day5_SW_TOTAL": 5,
    "day8_SW_TOTAL": 8,
})

with open(OUTDIR / "model_output.txt", "w") as f:
    # Separate OLS models by day: Total Distance
    for day_col, label in [("day5_Totaldistance", "Day 5 Total Distance"), ("day8_Totaldistance", "Day 8 Total Distance")]:
        model = smf.ols(f"{day_col} ~ C(genotype_label)", data=df).fit()
        write_block(f, f"OLS: {label} ~ genotype", model.summary().as_text())
        write_block(f, f"ANOVA: {label} ~ genotype", anova_lm(model).to_string())

    # Separate OLS models by day: SW/Total
    for day_col, label in [("day5_SW_TOTAL", "Day 5 SW/Total"), ("day8_SW_TOTAL", "Day 8 SW/Total")]:
        model = smf.ols(f"{day_col} ~ C(genotype_label)", data=df).fit()
        write_block(f, f"OLS: {label} ~ genotype", model.summary().as_text())
        write_block(f, f"ANOVA: {label} ~ genotype", anova_lm(model).to_string())

    # Repeated-measures / mixed models
    td_mixed = smf.mixedlm(
        "Total_Distance ~ C(genotype_label) + Day + Age",
        data=td_long,
        groups=td_long["mouse_ID"]
    ).fit(reml=False)
    write_block(f, "MixedLM: Total_Distance ~ genotype + Day + Age + (1|mouse_ID)", td_mixed.summary().as_text())

    td_mixed_int = smf.mixedlm(
        "Total_Distance ~ C(genotype_label) * Day + Age",
        data=td_long,
        groups=td_long["mouse_ID"]
    ).fit(reml=False)
    write_block(f, "MixedLM: Total_Distance ~ genotype * Day + Age + (1|mouse_ID)", td_mixed_int.summary().as_text())

    sw_mixed = smf.mixedlm(
        "SW_Total ~ C(genotype_label) * Day",
        data=sw_long,
        groups=sw_long["mouse_ID"]
    ).fit(reml=False)
    write_block(f, "MixedLM: SW_Total ~ genotype * Day + (1|mouse_ID)", sw_mixed.summary().as_text())

print(f"Saved model output to: {OUTDIR.resolve() / 'model_output.txt'}")
