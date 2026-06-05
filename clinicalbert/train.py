import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
import wandb

#=== Config =============================================#
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
DATA_PATH = "clinicalbert/data/sample.csv"
MAX_LEN = 128
BATCH_SIZE = 2
EPOCHS = 3
LR = 2e-4

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

#=== Data ===============================================#
dataset = ClinicalDataset(DATA_PATH, tokenizer, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#=== W&B ================================================#
wandb.init(
    entity="alzheimer-nlp",
    project="alzheimer-nlp",
    name="clinicalbert-lora-run1",
    config={"model": MODEL_NAME, "epochs": EPOCHS, "lr": LR, "lora_r": 8}
)

#=== Training Loop ======================================#
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0

    for batch in dataloader:
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

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataset)

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")
    wandb.log({"epoch": epoch+1, "loss": avg_loss, "accuracy": accuracy})

wandb.finish()
print("Done.")