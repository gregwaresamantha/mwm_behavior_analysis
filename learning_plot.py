"""
Created on 3/15/26
@author: samgregware
"""
if __name__ == '__main__':
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = pd.read_csv("mwm_learning_data.csv")
    df.columns = df.columns.str.strip()

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

    learning_long["Day"] = learning_long["Day"].str.extract("(\d)").astype(int)

    sns.lineplot(
        data=learning_long,
        x="Day",
        y="SW_distance",
        hue="Genotype",
        estimator="mean",
        errorbar="se"
    )

    plt.title("Learning Curve: SW Distance Across Training Days")
    plt.show()
