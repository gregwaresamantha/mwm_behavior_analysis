"""
Created on 3/15/26
@author: samgregware
"""

if __name__ == '__main__':
    import pandas as pd
    import statsmodels.formula.api as smf
    from pathlib import Path

    # -----------------------------
    # SETTINGS
    # -----------------------------
    DATA_FILE = "mwm_data.csv"
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)

    GENO_REF = "APOE33HN"
    SEX_REF = "F"

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    print("Columns in mwm_data.csv:")
    print(df.columns.tolist())

    # -----------------------------
    # FIND NEEDED COLUMNS
    # -----------------------------
    mouse_col = "Animal Code"
    geno_col = "Genotype"
    sex_col = "Sex"

    probe_day5_dist_col = "Distance Probe Day 5"
    probe_day8_dist_col = "Distance Probe Day 8"
    probe_day5_sw_col = "SW Distance Probe Day 5"
    probe_day8_sw_col = "SW Distance Probe Day 8"

    # -----------------------------
    # CLEAN STRINGS
    # -----------------------------
    df[mouse_col] = df[mouse_col].astype(str).str.strip()
    df[geno_col] = df[geno_col].astype(str).str.strip()
    df[sex_col] = df[sex_col].astype(str).str.strip()

    # -----------------------------
    # CLEAN NUMERIC PROBE COLUMNS
    # -----------------------------
    for col in [probe_day5_dist_col, probe_day8_dist_col, probe_day5_sw_col, probe_day8_sw_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------
    # COMPUTE SW/TOTAL
    # -----------------------------
    df["day5_SW_TOTAL"] = df[probe_day5_sw_col] / df[probe_day5_dist_col]
    df["day8_SW_TOTAL"] = df[probe_day8_sw_col] / df[probe_day8_dist_col]

    # -----------------------------
    # DAY 5 GLM
    # -----------------------------
    probe_day5_df = df.dropna(subset=["day5_SW_TOTAL"]).copy()

    formula_day5 = (
        f"day5_SW_TOTAL ~ "
        f"C({geno_col}, Treatment(reference='{GENO_REF}')) + "
        f"C({sex_col}, Treatment(reference='{SEX_REF}'))"
    )

    model_day5 = smf.ols(formula_day5, data=probe_day5_df).fit()

    print("\nPROBE DAY 5 GLM")
    print(model_day5.summary())

    results_day5 = pd.DataFrame({
        "term": model_day5.params.index,
        "estimate": model_day5.params.values,
        "std_error": model_day5.bse.values,
        "t_value": model_day5.tvalues.values,
        "p_value": model_day5.pvalues.values
    })

    results_day5["model"] = f"GLM: {formula_day5}"
    results_day5["N"] = len(probe_day5_df)

    results_day5.to_csv(RESULTS_DIR / "probe_day5_glm_results.csv", index=False)

    # -----------------------------
    # DAY 8 GLM
    # -----------------------------
    probe_day8_df = df.dropna(subset=["day8_SW_TOTAL"]).copy()

    formula_day8 = (
        f"day8_SW_TOTAL ~ "
        f"C({geno_col}, Treatment(reference='{GENO_REF}')) + "
        f"C({sex_col}, Treatment(reference='{SEX_REF}'))"
    )

    model_day8 = smf.ols(formula_day8, data=probe_day8_df).fit()

    print("\nPROBE DAY 8 GLM")
    print(model_day8.summary())

    results_day8 = pd.DataFrame({
        "term": model_day8.params.index,
        "estimate": model_day8.params.values,
        "std_error": model_day8.bse.values,
        "t_value": model_day8.tvalues.values,
        "p_value": model_day8.pvalues.values
    })

    results_day8["model"] = f"GLM: {formula_day8}"
    results_day8["N"] = len(probe_day8_df)

    results_day8.to_csv(RESULTS_DIR / "probe_day8_glm_results.csv", index=False)

    print("\nSaved:")
    print("- results/probe_day5_glm_results.csv")
    print("- results/probe_day8_glm_results.csv")