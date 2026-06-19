import pandas as pd
import re

IN_PATH  = "clinicalbert/data/mimic_dementia.csv"
OUT_PATH = "clinicalbert/data/mimic_dementia_masked.csv"

KEYWORD_PATTERN = re.compile(r"dementia|alzheimer\w*", re.IGNORECASE)

df = pd.read_csv(IN_PATH)

before_dementia = df[df["label"] == 1]["text"].str.contains(KEYWORD_PATTERN, na=False).sum()
before_control  = df[df["label"] == 0]["text"].str.contains(KEYWORD_PATTERN, na=False).sum()

print(f"Before masking:")
print(f"  Dementia notes with keyword: {before_dementia} / {(df['label']==1).sum()}")
print(f"  Control notes with keyword:  {before_control} / {(df['label']==0).sum()}")

df["text"] = df["text"].apply(lambda t: KEYWORD_PATTERN.sub("[MASKED]", t))

after_dementia = df[df["label"] == 1]["text"].str.contains(KEYWORD_PATTERN, na=False).sum()
print(f"\nAfter masking, dementia notes still containing keyword: {after_dementia} (should be 0)")

df.to_csv(OUT_PATH, index=False)
print(f"Saved {len(df)} masked samples to {OUT_PATH}")