import pandas as pd

# =========================
# 1. LOAD FILES
# =========================
mapping = pd.read_csv("subject_mapping.csv")
glp1 = pd.read_csv("GLP1metadata(Sheet1).csv")
meta = pd.read_csv("metadata.csv")

# =========================
# 2. CLEAN COLUMN NAMES
# =========================
for df in [mapping, glp1, meta]:
    df.columns = df.columns.str.strip()

# =========================
# 3. CLEAN BadeaID
# =========================
for df_name, df in [("mapping", mapping), ("glp1", glp1), ("meta", meta)]:
    if "BadeaID" not in df.columns:
        raise KeyError(f"{df_name} is missing 'BadeaID'")
    df["BadeaID"] = df["BadeaID"].astype(str).str.strip()

# =========================
# 4. MERGE subject_mapping WITH GLP1 BY BadeaID
# =========================
glp1_mapped = pd.merge(glp1, mapping, on="BadeaID", how="left")

# =========================
# 5. APPEND TO LARGE metadata.csv
#    (only add new BadeaIDs not already present)
# =========================
glp1_new = glp1_mapped[~glp1_mapped["BadeaID"].isin(meta["BadeaID"])]

combined = pd.concat([meta, glp1_new], ignore_index=True, sort=False)

# =========================
# 6. SAVE FILES
# =========================
glp1_mapped.to_csv("GLP1_with_subject_mapping.csv", index=False)
combined.to_csv("metadata_appended.csv", index=False)

# =========================
# 7. PRINT SUMMARY
# =========================
print("Done!")
print("Rows in subject_mapping:", len(mapping))
print("Rows in GLP1metadata:", len(glp1))
print("Rows after GLP1 + subject_mapping merge:", len(glp1_mapped))
print("Rows already in metadata:", len(meta))
print("New GLP1 rows added:", len(glp1_new))
print("Final total rows:", len(combined))
print("Final columns:", len(combined.columns))