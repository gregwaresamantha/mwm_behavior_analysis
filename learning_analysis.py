"""
Created on 3/15/26
@author: samgregware
"""
if __name__ == '__main__':
    import pandas as pd
    import statsmodels.formula.api as smf

    # 1) load csv
    df = pd.read_csv("mwm_learning_data.csv")

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
            "SW Day 1 Average Distance (m)",
            "SW Day 2 Average Distance (m)",
            "SW Day 3 Average Distance (m)",
            "SW Day 4 Average Distance (m)",
            "SW Day 5 Average Distance (m)",
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
            "SW Day 1 Average Distance (m)",
            "SW Day 2 Average Distance (m)",
            "SW Day 3 Average Distance (m)",
            "SW Day 4 Average Distance (m)",
            "SW Day 5 Average Distance (m)"
        ],
        var_name="Day",
        value_name="SW_distance"
    )

    # 8) clean long data
    learning_long = learning_long.dropna(subset=["SW_distance"])
    learning_long["SW_distance"] = pd.to_numeric(learning_long["SW_distance"],
                                                 errors="coerce")
    learning_long = learning_long.dropna(subset=["SW_distance"])

    learning_long["Day"] = learning_long["Day"].str.extract(r"(\d+)").astype(
        int)
    learning_long["MouseID"] = learning_long["Animal Code"].astype(
        str).str.strip()

    # remove any accidental blank IDs
    learning_long = learning_long[learning_long["MouseID"] != ""]

    # 9) set categories
    learning_long["Genotype"] = learning_long["Genotype"].astype("category")
    learning_long["Sex"] = learning_long["Sex"].astype("category")
    learning_long["MouseID"] = learning_long["MouseID"].astype(str)

    # 10) center day
    learning_long["Day_c"] = learning_long["Day"] - learning_long["Day"].mean()

    print(learning_long.head())
    print(learning_long["MouseID"].apply(type).value_counts())

    # 11) mixed model
    model = smf.mixedlm(
        "SW_distance ~ Day_c * Genotype + Day_c * Sex",
        data=learning_long,
        groups=learning_long["MouseID"]
    )

    result = model.fit(reml=False)

    print(result.summary())

    # 12) export results
    results_df = pd.DataFrame({
        "term": result.params.index,
        "estimate": result.params.values,
        "std_error": result.bse.values,
        "z_value": result.tvalues.values,
        "p_value": result.pvalues.values
    })

    results_df[
        "model"] = "MixedLM: SW_distance ~ Day_c * Genotype + Day_c * Sex + (1|MouseID)"
    results_df["N_mice"] = learning_long["MouseID"].nunique()
    results_df["N_obs"] = learning_long.shape[0]

    results_df.to_csv("learning_mixedlm_results.csv", index=False)
    print("Saved: learning_mixedlm_results.csv")