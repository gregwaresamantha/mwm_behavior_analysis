"""
Created on 5/8/26
@author: samgregware
"""
import pandas as pd

# Load the two CSV files
df1 = pd.read_csv("CLEAN_MWM_CONTROLS_single_row.csv")
df2 = pd.read_csv("MWM12302025.csv")

# Merge while keeping ALL columns from both sheets
merged_df = pd.concat([df1, df2], ignore_index=True, sort=False)

# Optional: remove duplicate rows
merged_df = merged_df.drop_duplicates()

# Save merged file
merged_df.to_csv("Merged_MWM_Combined.csv", index=False)

print("Files merged successfully!")
print(f"Rows: {merged_df.shape[0]}")
print(f"Columns: {merged_df.shape[1]}")

print("\nColumns in merged file:")
print(merged_df.columns.tolist())
