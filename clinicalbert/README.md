# Alzheimer's Early Detection from Clinical Notes (ClinicalBERT + LoRA)

A fine-tuned ClinicalBERT model that classifies dementia indicators in real clinical
discharge summaries from the MIMIC-III database, deployed as a containerized inference API.

## Motivation

This project is part of a three-part summer portfolio (`alzheimer-nlp`) focused on
healthcare AI for Alzheimer's and dementia detection, an area of personal motivation
stemming from childhood volunteering with dementia patients. This first project builds
the core NLP classification pipeline. Later projects in the series extend it into RAG
over medical literature and a full MLOps deployment pipeline.

## What it does

Given a piece of clinical text, the model predicts whether it indicates cognitive
decline/dementia or not, with a confidence score.

Request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient unable to recall recent events, disoriented to time and place."}'
```

Response:
```json
{
  "prediction": "dementia",
  "confidence": 0.81,
  "probabilities": {"control": 0.19, "dementia": 0.81}
}
```

## Architecture

1. Clinical text is tokenized using the Bio_ClinicalBERT tokenizer (max 512 tokens)
2. Tokens pass through a frozen ClinicalBERT encoder with LoRA adapters (rank 8) injected into the query/value attention matrices
3. A classification head (768 to 2) outputs logits for "control" vs. "dementia"
4. The model is served through a FastAPI endpoint, packaged in a Docker container

- **Base model:** [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT), BERT pretrained on MIMIC-III clinical text
- **Fine-tuning method:** LoRA via HuggingFace PEFT (0.27% of parameters trainable: 296K of 108.6M)
- **Experiment tracking:** Weights & Biases
- **Serving:** FastAPI + Docker

## Dataset

Built from the [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/).
Credentialed PhysioNet access is required to reproduce this; raw data is not redistributed
in this repo (see Data Access below).

**Construction:**
1. Identified patients with dementia-related ICD-9 codes (`290.x`) from `DIAGNOSES_ICD`
2. Pulled their discharge summaries from `NOTEEVENTS`, labeled `1` (dementia)
3. Sampled an equal number of discharge summaries from patients without a dementia code, labeled `0` (control)
4. Excluded all `NEWBORN`-type admissions (via `ADMISSIONS.ADMISSION_TYPE`) from both groups to avoid a trivial confound: without this filter, the model could learn to distinguish "elderly patient" vs. "newborn" rather than learning dementia-specific language

| | Count |
|---|---|
| Dementia notes | 293 |
| Control notes | 293 |
| **Total** | **586** |
| Mean note length | 1,571 words |
| Train / Val split | 469 / 117 (80/20, seeded) |

**Note on prevalence:** this is a balanced research subset, not representative of real-world
dementia prevalence in a hospital population. A production model would need to handle
significant class imbalance.

## Results

Best validation performance across training runs:

| Metric | Score |
|---|---|
| F1 | 0.87 |
| Precision | 0.85 |
| Recall | 0.89 |

**Run-to-run variance:** F1 ranged 0.81 to 0.87 across different training runs on identical
data and hyperparameters, due to LoRA weight initialization and training order randomness
on a relatively small (586-sample) dataset. This is expected at this data scale and would
stabilize with more training data.

Training used early stopping (patience of 3 epochs on validation F1) to avoid overfitting.
Train accuracy reached 95%+ while validation F1 plateaued, a clear memorization signal on
a dataset this size.

## Known limitation: distribution sensitivity

The model was evaluated against three manually constructed test cases beyond the
validation set:

| Input | Style | Prediction | Confidence | Correct? |
|---|---|---|---|---|
| "Patient unable to recall recent events, disoriented to time and place." | Short, explicit dementia language | dementia | 0.81 | Yes |
| "Patient alert and oriented... no cognitive complaints... independent in ADLs." | Short, explicit control language | dementia | 0.64 | No |
| Full MIMIC-style discharge summary (admission/discharge dates, "History of present illness" header, ~70 words) | Long-form, matches training distribution | control | 0.58 | Yes |

**Finding:** the model performs reliably on long-form text resembling MIMIC-III discharge
summaries (its training distribution) but is unreliable on short, structured clinical
statements that don't resemble that format. Training data was exclusively full discharge
summaries (mean 1,571 words), so a 20-word sentence is out-of-distribution input.

This is disclosed here deliberately rather than hidden. Understanding a model's failure
modes is as important as reporting its headline metrics.

**Future work:** expand training data to include shorter-form clinical assessments
(e.g. cognitive screening notes) to improve robustness across note lengths and styles.

## Ablation: reliance on explicit diagnosis keywords

85.7% of dementia-labeled notes contain the literal word "dementia" or "alzheimer"
somewhere in the text (often in a discharge diagnosis line). This raised a question:
is the model learning to detect cognitive decline from symptom language, or is it
mostly pattern-matching the diagnostic keyword itself? These are different tasks of
very different difficulty, and only the first is a meaningful step toward early
detection.

To test this, the keyword was masked (`dementia` / `alzheimer*` replaced with
`[MASKED]`) in all notes, and the model was retrained from scratch on the masked
dataset, with identical hyperparameters and the same train/val split.

| Version | Val F1 | Precision | Recall |
|---|---|---|---|
| Original (keyword present) | 0.87 | 0.85 | 0.89 |
| Keyword masked | 0.80 | 0.76 | 0.84 |

F1 dropped by roughly 7 points after masking, not a collapse to chance level (0.50).
This indicates the model relies on the explicit keyword to some degree, but is also
learning real symptom-language signal independent of it. Supporting this: 21 of 293
control notes also contained "dementia" or "alzheimer" (e.g. ruled-out differential
diagnoses, family history mentions), so the keyword was never a perfect shortcut to
the label in the first place.

**Takeaway:** the reported 0.87 F1 reflects a task that includes some diagnostic-label
leakage. The 0.80 F1 under masking is a more honest estimate of the model's ability
to detect dementia-related language without relying on an explicit diagnosis being
stated, closer to what an early-detection system would need.

Full run logged on [Weights & Biases](https://wandb.ai/alzheimer-nlp/alzheimer-nlp)
under the run name `clinicalbert-lora-masked-ablation`.

## Project structure

```
clinicalbert/
├── data/
│   ├── build_mimic_dataset.py    # Filters MIMIC-III for dementia/control discharge summaries
│   └── mimic_dementia.csv        # Processed dataset (not redistributed, see Data Access)
├── train.py                      # LoRA fine-tuning with early stopping and W&B logging
├── evaluate.py                   # Standalone evaluation on saved checkpoint
├── checkpoints/best_model/       # Best LoRA checkpoint (by validation F1)
└── api/
    ├── main.py                   # FastAPI inference endpoint
    └── Dockerfile
```

## Running it

Supports CUDA, Apple Silicon (MPS), and CPU automatically via runtime device detection.
No code changes needed across environments.

### Local (Python)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train (requires MIMIC-III access, see Data Access below)
python clinicalbert/train.py

# Evaluate the saved checkpoint
python clinicalbert/evaluate.py

# Serve the API
uvicorn clinicalbert.api.main:app --reload --port 8000
```

### Docker

```bash
docker build -t alzheimer-nlp-api -f clinicalbert/api/Dockerfile .
docker run -p 8000:8000 alzheimer-nlp-api
```

Interactive API docs are available at `http://localhost:8000/docs` once running.

## Data access

This repo does not include raw MIMIC-III data or the processed dataset, in compliance
with the PhysioNet Data Use Agreement. To reproduce:

1. Complete CITI "Data or Specimens Only Research" training
2. Apply for credentialed access at [physionet.org](https://physionet.org/content/mimiciii/1.4/)
3. Download `NOTEEVENTS.csv.gz`, `DIAGNOSES_ICD.csv.gz`, `ADMISSIONS.csv.gz`
4. Run `python clinicalbert/data/build_mimic_dataset.py`

## Tech stack

PyTorch, HuggingFace Transformers, PEFT (LoRA), Weights & Biases, FastAPI, Docker, scikit-learn