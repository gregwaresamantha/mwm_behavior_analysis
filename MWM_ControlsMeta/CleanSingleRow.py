import pandas as pd

# Load the new sheet
df = pd.read_csv("CLEAN_MWM_CONTROLS.csv")

# Extract Day and Trial from Row
# Examples:
# 190610_10_Day1_T1_positions.csv
# 30605_9_Probe_D5_T1_positions.csv

df["IsProbe"] = df["Row"].str.contains("Probe", case=False, na=False)

df["Day"] = df["Row"].str.extract(r"(?:Day|D)(\d+)", expand=False)
df["Trial"] = df["Row"].str.extract(r"T(\d+)", expand=False)

df = df.dropna(subset=["Day", "Trial"])

df["Day"] = df["Day"].astype(int)
df["Trial"] = df["Trial"].astype(int)

df["Phase"] = df["IsProbe"].map({True: "Probe", False: ""})

# Mouse/info columns to keep at the front
id_cols = [
    "Animal Code",
    "CohortID",
    "Treatment",
    "Diet",
    "Sex",
    "Genotype",
    "Age_mastersheet",
    "Lifestyle"
]

# Measurement columns to spread horizontally
measurement_cols = [
    "Time",
    "Duration",
    "Distance",
    "MeanSpeed",
    "NE_Time",
    "NE_Distance",
    "NE_AverageSpeed",
    "NW_Time",
    "NW_Distance",
    "NW_AverageSpeed",
    "SE_Time",
    "SE_Distance",
    "SE_AverageSpeed",
    "SW_Time",
    "SW_Distance",
    "SW_AverageSpeed",
    "Island_Entries",
    "Island_NumberExits",
    "Island_Time",
    "Island_DistanceToFirstEntry",
    "Island_LatencyToFirstEntry",
    "NormSWTime",
    "NormSWDist",
    "DistTot",
    "SW.Dist.Norm",
    "TimeTot",
    "SE.Time.Norm",
    "SW.Time.Norm"
]

# Only keep columns that exist
id_cols = [col for col in id_cols if col in df.columns]
measurement_cols = [col for col in measurement_cols if col in df.columns]

# Pivot to one row per mouse
wide_df = df.pivot_table(
    index=id_cols,
    columns=["Phase", "Day", "Trial"],
    values=measurement_cols,
    aggfunc="first"
)

# Flatten column names
wide_df.columns = [
    f"{measure}{phase}Day{day}Trial{trial}"
    for measure, phase, day, trial in wide_df.columns
]

wide_df = wide_df.reset_index()

# Save output
wide_df.to_csv("CLEAN_MWM_CONTROLS_single_row.csv", index=False)

print("Done!")
print(wide_df.shape)
print(wide_df.head())