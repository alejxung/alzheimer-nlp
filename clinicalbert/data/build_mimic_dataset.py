import pandas as pd

RAW_DIR  = "clinicalbert/data/mimic_raw"
OUT_PATH = "clinicalbert/data/mimic_dementia.csv"

# Find dementia HADM_IDs
print("Loading diagnoses...")
diagnoses = pd.read_csv(
    f"{RAW_DIR}/DIAGNOSES_ICD.csv.gz",
    usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
    dtype={"ICD9_CODE": str}
)

dementia_mask = diagnoses["ICD9_CODE"].str.startswith("290", na=False)
dementia_hadm_ids = set(diagnoses.loc[dementia_mask, "HADM_ID"].unique())
print(f"Dementia HADM_IDs found: {len(dementia_hadm_ids)}")

# Load admissions and exclude newborns
print("Loading admissions...")
admissions = pd.read_csv(
    f"{RAW_DIR}/ADMISSIONS.csv.gz",
    usecols=["SUBJECT_ID", "HADM_ID", "ADMISSION_TYPE"]
)

adult_hadm_ids = set(
    admissions.loc[admissions["ADMISSION_TYPE"] != "NEWBORN", "HADM_ID"].unique()
)
print(f"Non-newborn HADM_IDs: {len(adult_hadm_ids)}")

all_hadm_ids = set(diagnoses["HADM_ID"].unique())

# Controls: adult admissions that are not dementia
control_hadm_ids = (all_hadm_ids & adult_hadm_ids) - dementia_hadm_ids
print(f"Control (adult, non-dementia) HADM_IDs available: {len(control_hadm_ids)}")

# Restrict dementia group to adult admissions
dementia_hadm_ids = dementia_hadm_ids & adult_hadm_ids
print(f"Dementia HADM_IDs after adult filter: {len(dementia_hadm_ids)}")

# Load discharge summaries
print("Loading notes (this takes a while)...")
notes = pd.read_csv(
    f"{RAW_DIR}/NOTEEVENTS.csv.gz",
    usecols=["SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT"],
    low_memory=False
)

discharge_notes = notes[notes["CATEGORY"] == "Discharge summary"].copy()
discharge_notes = discharge_notes.dropna(subset=["HADM_ID"])
discharge_notes["HADM_ID"] = discharge_notes["HADM_ID"].astype(int)
print(f"Total discharge summaries: {len(discharge_notes)}")

# Label notes and build dataset
dementia_notes = discharge_notes[
    discharge_notes["HADM_ID"].isin(dementia_hadm_ids)
].copy()
dementia_notes["label"] = 1

control_notes = discharge_notes[
    discharge_notes["HADM_ID"].isin(control_hadm_ids)
].copy()
control_notes["label"] = 0

n_dementia = len(dementia_notes)
control_notes = control_notes.sample(
    n=min(n_dementia, len(control_notes)),
    random_state=42
)

print(
    f"Dementia notes: {len(dementia_notes)} | Control notes: {len(control_notes)}"
)

# Combine, shuffle, and save
final_df = pd.concat([dementia_notes, control_notes])[["TEXT", "label"]]
final_df = final_df.rename(columns={"TEXT": "text"})
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

final_df.to_csv(OUT_PATH, index=False)
print(f"Saved {len(final_df)} samples to {OUT_PATH}")
