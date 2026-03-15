"""
Created on 3/15/26
@author: samgregware
"""
if __name__ == '__main__':
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.read_csv("mwm_probe_data.csv")
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # clean strings
    df["genotype"] = df["genotype"].astype(str).str.strip()
    df["sex"] = df["sex"].astype(str).str.strip()

    # if you have a diet column, keep this model;
    # if not, use the simpler model below
    formula = "day5_SW_TOTAL ~ genotype + sex"

    model = smf.ols(formula, data=df).fit()

    print(model.summary())

    results_df = pd.DataFrame({
        "term": model.params.index,
        "estimate": model.params.values,
        "std_error": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values
    })

    results_df["model"] = f"GLM: {formula}"
    results_df["N"] = len(df)

    results_df.to_csv("probe_glm_results.csv", index=False)
    print("Saved: probe_glm_results.csv")
