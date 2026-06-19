import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import wandb
import os

#=== Config =============================================#
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
SPLIT_DIR = "clinicalbert/data/splits"
MAX_LEN = 512
BATCH_SIZE = 4
EPOCHS = 15
LR = 2e-4
PATIENCE = 3
CHECKPOINT_DIR = "clinicalbert/checkpoints"

#=== Device ==============================================#
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

#=== Model + LoRA =======================================#
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = model.to(device)

#=== Load persisted splits ================================#
if not os.path.exists(f"{SPLIT_DIR}/train.csv"):
    raise FileNotFoundError(
        f"No split files found at {SPLIT_DIR}/. "
        f"Run `python clinicalbert/data/split_dataset.py` first."
    )

train_dataset = ClinicalDataset(f"{SPLIT_DIR}/train.csv", tokenizer, MAX_LEN)
val_dataset = ClinicalDataset(f"{SPLIT_DIR}/val.csv", tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
print("(Test set held out separately, not loaded during training)")

#=== W&B ================================================#
wandb.init(
    project="alzheimer-nlp",
    name="clinicalbert-lora-run5-3way-split",
    config={
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "lr": LR,
        "lora_r": 8,
        "batch_size": BATCH_SIZE,
        "max_len": MAX_LEN,
        "device": str(device),
        "patience": PATIENCE,
        "split_scheme": "70/15/15 stratified, persisted to disk"
    }
)

#=== Eval function ======================================#
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "labels": all_labels,
        "preds": all_preds
    }

#=== Training loop with early stopping ==================#
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
best_val_f1 = 0.0
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc = correct / len(train_dataset)

    val_metrics = evaluate(model, val_loader)

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val F1: {val_metrics['f1']:.4f} | "
        f"Val Precision: {val_metrics['precision']:.4f} | "
        f"Val Recall: {val_metrics['recall']:.4f}"
    )
    print(f"Confusion Matrix:\n{val_metrics['confusion_matrix']}\n")

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_f1": val_metrics["f1"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_confusion_matrix": wandb.plot.confusion_matrix(
            y_true=val_metrics["labels"],
            preds=val_metrics["preds"],
            class_names=["control", "dementia"]
        )
    })

    if val_metrics["f1"] > best_val_f1:
        best_val_f1 = val_metrics["f1"]
        epochs_without_improvement = 0
        save_path = os.path.join(CHECKPOINT_DIR, "best_model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"  -> New best val F1: {best_val_f1:.4f}. Saved to {save_path}\n")
    else:
        epochs_without_improvement += 1
        print(f"  -> No improvement for {epochs_without_improvement} epoch(s).\n")

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}. Best val F1: {best_val_f1:.4f}")
            break

wandb.finish()
print(f"Done. Best val F1: {best_val_f1:.4f}")
print("\nNOTE: this is the VALIDATION F1, used for model selection.")
print("Run `python clinicalbert/evaluate.py --split test` ONCE for the unbiased test metric.")