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

print("Columns in mwm_data.csv:")
print(df.columns.tolist())

# -----------------------------
# HELPERS
# -----------------------------
def find_col(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    raise KeyError(f"None of these columns were found: {possible_names}")

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

def export_glm(result, label, outfile, n):
    out = pd.DataFrame({
        "term": result.params.index,
        "estimate": result.params.values,
        "std_error": result.bse.values,
        "t_value": result.tvalues.values,
        "p_value": result.pvalues.values
    })
    out["model"] = label
    out["N"] = n
    out.to_csv(RESULTS_DIR / outfile, index=False)

# -----------------------------
# CORE COLUMN NAMES
# -----------------------------
mouse_col = find_col(["Animal Code", "Animal_ID", "mouse_ID", "MouseID"])
geno_col = find_col(["Genotype", "genotype"])
sex_col = find_col(["Sex", "sex"])

# clean id/factors
df[mouse_col] = df[mouse_col].astype(str).str.strip()
df[geno_col] = df[geno_col].astype(str).str.strip()
df[sex_col] = df[sex_col].astype(str).str.strip()

# -----------------------------
# LEARNING: TOTAL DISTANCE MIXED MODEL
# -----------------------------
day1_total = find_col(["Day1_Totaldistance", "day1_Totaldistance", "Day 1 Average (m)"])
day2_total = find_col(["Day2_Totaldistance", "day2_Totaldistance", "Day 2 Average (m)"])
day3_total = find_col(["Day3_Totaldistance", "day3_Totaldistance", "Day 3 Average (m)"])
day4_total = find_col(["Day4_Totaldistance", "day4_Totaldistance", "Day 4 Average (m)"])
day5_total = find_col(["Day5_Totaldistance", "day5_Totaldistance", "Day 5 Average (m)"])

learning_total = df[[mouse_col, geno_col, sex_col, day1_total, day2_total, day3_total, day4_total, day5_total]].copy()

learning_total_long = learning_total.melt(
    id_vars=[mouse_col, geno_col, sex_col],
    value_vars=[day1_total, day2_total, day3_total, day4_total, day5_total],
    var_name="Day",
    value_name="TotalDistance"
)

learning_total_long["Day"] = (
    learning_total_long["Day"]
    .str.extract(r"(\d+)")
    .astype(int)
)
learning_total_long["Day_c"] = learning_total_long["Day"] - learning_total_long["Day"].mean()

learning_total_long["TotalDistance"] = pd.to_numeric(learning_total_long["TotalDistance"], errors="coerce")
learning_total_long = learning_total_long.dropna(subset=["TotalDistance"]).copy()

learning_total_long["MouseID"] = learning_total_long[mouse_col].astype("category")
learning_total_long["Genotype"] = learning_total_long[geno_col].astype("category")
learning_total_long["Sex"] = learning_total_long[sex_col].astype("category")

total_formula = (
    f"TotalDistance ~ Day_c * C(Genotype, Treatment(reference='{GENO_REF}')) "
    f"+ Day_c * C(Sex, Treatment(reference='{SEX_REF}'))"
)

total_model = smf.mixedlm(
    total_formula,
    data=learning_total_long,
    groups=learning_total_long["MouseID"]
).fit(reml=False)

print("\nLEARNING TOTAL DISTANCE MIXED MODEL")
print(total_model.summary())

export_mixedlm(
    total_model,
    f"MixedLM: {total_formula} + (1|MouseID)",
    "learning_totaldistance_mixedlm_results.csv",
    learning_total_long["MouseID"].nunique(),
    learning_total_long.shape[0]
)

# -----------------------------
# LEARNING: SW DISTANCE MIXED MODEL
# -----------------------------
day1_sw = find_col(["Day1_SW"])
day2_sw = find_col(["Day2_SW"])
day3_sw = find_col(["Day3_SW"])
day4_sw = find_col(["Day4_SW"])
day5_sw = find_col(["Day5_SW"])

learning_sw = df[[mouse_col, geno_col, sex_col, day1_sw, day2_sw, day3_sw, day4_sw, day5_sw]].copy()

learning_sw_long = learning_sw.melt(
    id_vars=[mouse_col, geno_col, sex_col],
    value_vars=[day1_sw, day2_sw, day3_sw, day4_sw, day5_sw],
    var_name="Day",
    value_name="SW_distance"
)

learning_sw_long["Day"] = (
    learning_sw_long["Day"]
    .str.extract(r"(\d+)")
    .astype(int)
)
learning_sw_long["Day_c"] = learning_sw_long["Day"] - learning_sw_long["Day"].mean()

learning_sw_long["SW_distance"] = pd.to_numeric(learning_sw_long["SW_distance"], errors="coerce")
learning_sw_long = learning_sw_long.dropna(subset=["SW_distance"]).copy()

learning_sw_long["MouseID"] = learning_sw_long[mouse_col].astype("category")
learning_sw_long["Genotype"] = learning_sw_long[geno_col].astype("category")
learning_sw_long["Sex"] = learning_sw_long[sex_col].astype("category")

sw_formula = (
    f"SW_distance ~ Day_c * C(Genotype, Treatment(reference='{GENO_REF}')) "
    f"+ Day_c * C(Sex, Treatment(reference='{SEX_REF}'))"
)

sw_model = smf.mixedlm(
    sw_formula,
    data=learning_sw_long,
    groups=learning_sw_long["MouseID"]
).fit(reml=False)

print("\nLEARNING SW DISTANCE MIXED MODEL")
print(sw_model.summary())

export_mixedlm(
    sw_model,
    f"MixedLM: {sw_formula} + (1|MouseID)",
    "learning_swdistance_mixedlm_results.csv",
    learning_sw_long["MouseID"].nunique(),
    learning_sw_long.shape[0]
)

# -----------------------------
# PROBE GLM: COMPUTE SW/TOTAL FIRST
# -----------------------------
probe_day5_dist = find_col(["Distance Probe Day 5"])
probe_day8_dist = find_col(["Distance Probe Day 8"])
probe_day5_sw = find_col(["SW Distance Probe Day 5"])
probe_day8_sw = find_col(["SW Distance Probe Day 8"])

probe_df = df[[
    mouse_col, geno_col, sex_col,
    probe_day5_dist, probe_day8_dist,
    probe_day5_sw, probe_day8_sw
]].copy()

for col in [probe_day5_dist, probe_day8_dist, probe_day5_sw, probe_day8_sw]:
    probe_df[col] = pd.to_numeric(probe_df[col], errors="coerce")

probe_df["day5_SW_TOTAL"] = probe_df[probe_day5_sw] / probe_df[probe_day5_dist]
probe_df["day8_SW_TOTAL"] = probe_df[probe_day8_sw] / probe_df[probe_day8_dist]

probe_day5_df = probe_df.dropna(subset=["day5_SW_TOTAL"]).copy()
probe_day8_df = probe_df.dropna(subset=["day8_SW_TOTAL"]).copy()

probe5_formula = (
    f"day5_SW_TOTAL ~ C({geno_col}, Treatment(reference='{GENO_REF}')) "
    f"+ C({sex_col}, Treatment(reference='{SEX_REF}'))"
)

probe8_formula = (
    f"day8_SW_TOTAL ~ C({geno_col}, Treatment(reference='{GENO_REF}')) "
    f"+ C({sex_col}, Treatment(reference='{SEX_REF}'))"
)

probe5_model = smf.ols(probe5_formula, data=probe_day5_df).fit()
probe8_model = smf.ols(probe8_formula, data=probe_day8_df).fit()

print("\nPROBE DAY 5 GLM")
print(probe5_model.summary())

print("\nPROBE DAY 8 GLM")
print(probe8_model.summary())

export_glm(
    probe5_model,
    f"GLM: {probe5_formula}",
    "probe_day5_glm_results.csv",
    len(probe_day5_df)
)

export_glm(
    probe8_model,
    f"GLM: {probe8_formula}",
    "probe_day8_glm_results.csv",
    len(probe_day8_df)
)

print("\nSaved:")
print("- results/learning_totaldistance_mixedlm_results.csv")
print("- results/learning_swdistance_mixedlm_results.csv")
print("- results/probe_day5_glm_results.csv")
print("- results/probe_day8_glm_results.csv")