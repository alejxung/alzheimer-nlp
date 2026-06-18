import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import wandb

#=== Config =============================================#
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
DATA_PATH = "clinicalbert/data/sample.csv"
MAX_LEN = 128
BATCH_SIZE = 4
EPOCHS = 10
LR = 2e-4
VAL_SPLIT = 0.2

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

#=== Train/Val Split ====================================#
full_dataset = ClinicalDataset(DATA_PATH, tokenizer, MAX_LEN)
val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {train_size} samples | Val: {val_size} samples")

#=== W&B ================================================#
wandb.init(
    project="alzheimer-nlp",
    name="clinicalbert-lora-run2",
    config={
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "lr": LR,
        "lora_r": 8,
        "batch_size": BATCH_SIZE,
        "val_split": VAL_SPLIT
    }
)

#=== Eval function ======================================#
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].tolist())

    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {"f1": f1, "precision": precision, "recall": recall, "confusion_matrix": cm}

#=== Training loop ======================================#
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0

    for batch in train_loader:
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
    train_acc = correct / train_size

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
        "val_recall": val_metrics["recall"]
    })

wandb.finish()
print("Done.")