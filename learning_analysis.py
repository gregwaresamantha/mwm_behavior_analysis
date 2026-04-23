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
    # 1) load csv
    # -----------------------------
    df = pd.read_csv(DATA_FILE)

    # 2) clean column names
    df.columns = df.columns.str.strip()

    # 3) remove empty Excel columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # 4) keep only needed columns
    df = df[
        [
            "Animal Code",
            "Genotype",
            "Sex",
            "Age (months)",
            "Day1_SW",
            "Day2_SW",
            "Day3_SW",
            "Day4_SW",
            "Day5_SW",
        ]
    ].copy()

    # 5) drop rows missing key identifiers
    df = df.dropna(subset=["Animal Code", "Genotype", "Sex"])

    # 6) make identifiers plain strings
    df["Animal Code"] = df["Animal Code"].astype(str).str.strip()
    df["Genotype"] = df["Genotype"].astype(str).str.strip()
    df["Sex"] = df["Sex"].astype(str).str.strip()

    # 7) wide -> long
    learning_long = df.melt(
        id_vars=["Animal Code", "Genotype", "Sex", "Age (months)"],
        value_vars=[
            "Day1_SW",
            "Day2_SW",
            "Day3_SW",
            "Day4_SW",
            "Day5_SW"
        ],
        var_name="Day",
        value_name="SW_distance"
    )

    # 8) clean long data
    learning_long["SW_distance"] = pd.to_numeric(
        learning_long["SW_distance"],
        errors="coerce"
    )
    learning_long = learning_long.dropna(subset=["SW_distance"]).copy()

    learning_long["Day"] = learning_long["Day"].str.extract(r"(\d+)").astype(int)
    learning_long["MouseID"] = learning_long["Animal Code"].astype(str).str.strip()

    # remove any accidental blank IDs
    learning_long = learning_long[learning_long["MouseID"] != ""]

    # 9) set categories
    learning_long["Genotype"] = learning_long["Genotype"].astype("category")
    learning_long["Sex"] = learning_long["Sex"].astype("category")
    learning_long["MouseID"] = learning_long["MouseID"].astype("category")

    # 10) center day
    learning_long["Day_c"] = learning_long["Day"] - learning_long["Day"].mean()

    print(learning_long.head())
    print("\nN mice:", learning_long["MouseID"].nunique())
    print("N observations:", learning_long.shape[0])

    # 11) mixed model
    formula = (
        f"SW_distance ~ "
        f"Day_c * C(Genotype, Treatment(reference='{GENO_REF}')) + "
        f"Day_c * C(Sex, Treatment(reference='{SEX_REF}'))"
    )

    model = smf.mixedlm(
        formula,
        data=learning_long,
        groups=learning_long["MouseID"]
    )

    result = model.fit(reml=False)

    print("\nLEARNING MIXED MODEL")
    print(result.summary())

    # 12) export results
    results_df = pd.DataFrame({
        "term": result.params.index,
        "estimate": result.params.values,
        "std_error": result.bse.values,
        "z_value": result.tvalues.values,
        "p_value": result.pvalues.values
    })

    results_df["model"] = f"MixedLM: {formula} + (1|MouseID)"
    results_df["N_mice"] = learning_long["MouseID"].nunique()
    results_df["N_obs"] = learning_long.shape[0]

    results_df.to_csv(RESULTS_DIR / "learning_mixedlm_results.csv", index=False)
    print("\nSaved: results/learning_mixedlm_results.csv")