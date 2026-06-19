import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

#=== Config =============================================#
DATA_PATH = "clinicalbert/data/mimic_dementia.csv"
MAX_LEN = 512
BATCH_SIZE = 4
VAL_SPLIT = 0.2
SEED = 42
CHECKPOINT_DIR = "clinicalbert/checkpoints/best_model"

#=== Device ===============================================#
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

#=== Dataset ============================================#
class ClinicalDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        encoded = self.tokenizer(
            row["text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["label"], dtype=torch.long)
        }

#=== Load saved model + tokenizer =========================#
print(f"Loading checkpoint from {CHECKPOINT_DIR}...")

peft_config = PeftConfig.from_pretrained(CHECKPOINT_DIR)
base_model = AutoModelForSequenceClassification.from_pretrained(
    peft_config.base_model_name_or_path, num_labels=2
)
model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

model = model.to(device)
model.eval()

#=== Reconstruct identical val split ======================#
full_dataset = ClinicalDataset(DATA_PATH, tokenizer, MAX_LEN)
val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size

generator = torch.Generator().manual_seed(SEED)
_, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Evaluating on {val_size} validation samples\n")

#=== Run inference =========================================#
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        preds = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch["labels"].cpu().tolist())

#=== Report ================================================#
f1 = f1_score(all_labels, all_preds, zero_division=0)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

print("=" * 50)
print("FINAL EVALUATION RESULTS")
print("=" * 50)
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"\nConfusion Matrix:")
print(f"                Predicted Control  Predicted Dementia")
print(f"Actual Control       {cm[0][0]:>4}              {cm[0][1]:>4}")
print(f"Actual Dementia      {cm[1][0]:>4}              {cm[1][1]:>4}")
print(f"\nFull Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["control", "dementia"]))