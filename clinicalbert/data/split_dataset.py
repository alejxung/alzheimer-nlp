import pandas as pd
from sklearn.model_selection import train_test_split
import os

#=== Config =============================================#
IN_PATH = "clinicalbert/data/mimic_dementia.csv"
SPLIT_DIR = "clinicalbert/data/splits"
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42

#=== Split ==============================================#
os.makedirs(SPLIT_DIR, exist_ok=True)

df = pd.read_csv(IN_PATH)

train_df, temp_df = train_test_split(
    df,
    test_size=(VAL_FRAC + TEST_FRAC),
    stratify=df["label"],
    random_state=SEED
)

relative_test_frac = TEST_FRAC / (VAL_FRAC + TEST_FRAC)
val_df, test_df = train_test_split(
    temp_df,
    test_size=relative_test_frac,
    stratify=temp_df["label"],
    random_state=SEED
)

train_df.to_csv(f"{SPLIT_DIR}/train.csv", index=False)
val_df.to_csv(f"{SPLIT_DIR}/val.csv", index=False)
test_df.to_csv(f"{SPLIT_DIR}/test.csv", index=False)

def summarize(name, d):
    n_dem = (d["label"] == 1).sum()
    n_ctrl = (d["label"] == 0).sum()
    print(f"{name:6s}: {len(d):3d} samples  (dementia: {n_dem}, control: {n_ctrl})")

print("Stratified split complete (seed={}):\n".format(SEED))
summarize("Train", train_df)
summarize("Val", val_df)
summarize("Test", test_df)
print(f"\nSplits saved to {SPLIT_DIR}/")
print("Test set is now fixed on disk. Do not touch it until final reporting.")