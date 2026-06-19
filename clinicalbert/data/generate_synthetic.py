# clinicalbert/data/generate_synthetic.py
import pandas as pd

data = {
    "text": [
        "Patient demonstrates significant word-finding difficulty and short-term memory loss.",
        "No signs of cognitive impairment. Patient alert and oriented.",
        "Caregiver reports increased confusion and repetitive questioning.",
        "Normal cognitive function. Mini-mental state exam within normal limits.",
        "Patient unable to recall recent events. Disoriented to time and place.",
        "Patient scored 24/30 on MMSE. Mild forgetfulness noted.",
        "No memory complaints. Independent in all activities of daily living.",
        "Progressive decline in language and executive function observed.",
        "Patient recognizes family members. Follows complex instructions.",
        "Significant short-term memory impairment. Requires supervision.",
        "Alert and cooperative. No evidence of cognitive decline.",
        "Difficulty managing finances and medications independently.",
        "Patient oriented to person, place, and time. Normal recall.",
        "Wandering behavior reported. Unable to recall home address.",
        "Cognitive assessment within normal limits for age.",
        "Patient repeats questions. Cannot recall breakfast this morning.",
        "Lives independently. Drives and manages own appointments.",
        "Confusion worsened at night. Sundowning behavior suspected.",
        "Language fluency intact. Follows three-step commands correctly.",
        "Word substitution errors noted. Cannot name common objects.",
    ],
    "label": [1,0,1,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
}

df = pd.DataFrame(data)
df.to_csv("clinicalbert/data/sample.csv", index=False)
print(f"Generated {len(df)} samples")