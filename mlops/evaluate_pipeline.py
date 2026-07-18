import os
import json
import mlflow
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from sklearn.metrics import f1_score, precision_score, recall_score
from dotenv import load_dotenv

load_dotenv()

#=== Config =================================================#
SPLIT_DIR = "clinicalbert/data/splits"
MAX_LEN = 512
BATCH_SIZE = 4
CHALLENGER_DIR = "mlops/checkpoints/best_model"
CHAMPION_DIR = "mlops/checkpoints/production_model"
BASELINE_DIR = "clinicalbert/checkpoints/best_model"
MLFLOW_TRACKING_URI = "http://localhost:5001"
MLFLOW_EXPERIMENT = "clinicalbert-dementia-classifier"
PROMOTION_THRESHOLD = 0.01 

#=== Device =================================================#
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

#=== Dataset ================================================#
class ClinicalDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len):
        self.df = __import__("pandas").read_csv(csv_path)
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

#=== Load model =============================================#
def load_model(checkpoint_dir):
    print(f"Loading model from {checkpoint_dir}...")
    peft_config = PeftConfig.from_pretrained(checkpoint_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path, num_labels=2
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = model.to(device)
    model.eval()
    return model, tokenizer

#=== Evaluate on test set ===================================#
def evaluate_on_test(model, tokenizer):
    dataset = ClinicalDataset(f"{SPLIT_DIR}/test.csv", tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

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

    return {
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "n_samples": len(all_labels)
    }

#=== Promote challenger to production =======================#
def promote_challenger():
    import shutil
    if os.path.exists(CHAMPION_DIR):
        shutil.rmtree(CHAMPION_DIR)
    shutil.copytree(CHALLENGER_DIR, CHAMPION_DIR)
    print(f"Challenger promoted to production at {CHAMPION_DIR}")

#=== Main ===================================================#
def run_evaluation():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # Load challenger (newly trained model)
    if not os.path.exists(CHALLENGER_DIR):
        raise FileNotFoundError(
            f"No challenger model found at {CHALLENGER_DIR}. "
            "Run `python mlops/train_pipeline.py` first."
        )

    challenger_model, challenger_tokenizer = load_model(CHALLENGER_DIR)
    challenger_metrics = evaluate_on_test(challenger_model, challenger_tokenizer)
    print(f"\nChallenger metrics (test set):")
    print(f"  F1:        {challenger_metrics['f1']:.4f}")
    print(f"  Precision: {challenger_metrics['precision']:.4f}")
    print(f"  Recall:    {challenger_metrics['recall']:.4f}")

    # Load champion (production model or baseline fallback)
    champion_dir = CHAMPION_DIR if os.path.exists(CHAMPION_DIR) else BASELINE_DIR
    print(f"\nChampion source: {champion_dir}")

    champion_model, champion_tokenizer = load_model(champion_dir)
    champion_metrics = evaluate_on_test(champion_model, champion_tokenizer)
    print(f"\nChampion metrics (test set):")
    print(f"  F1:        {champion_metrics['f1']:.4f}")
    print(f"  Precision: {champion_metrics['precision']:.4f}")
    print(f"  Recall:    {champion_metrics['recall']:.4f}")

    # Compare and decide
    f1_delta = challenger_metrics["f1"] - champion_metrics["f1"]
    promoted = f1_delta >= PROMOTION_THRESHOLD

    print(f"\nF1 delta: {f1_delta:+.4f} (threshold: {PROMOTION_THRESHOLD:+.4f})")
    print(f"Decision: {'PROMOTE challenger' if promoted else 'KEEP champion'}")

    # Log comparison to MLflow
    with mlflow.start_run(run_name="model-comparison"):
        mlflow.log_metrics({
            "challenger_f1": challenger_metrics["f1"],
            "challenger_precision": challenger_metrics["precision"],
            "challenger_recall": challenger_metrics["recall"],
            "champion_f1": champion_metrics["f1"],
            "champion_precision": champion_metrics["precision"],
            "champion_recall": champion_metrics["recall"],
            "f1_delta": f1_delta
        })
        mlflow.log_params({
            "promotion_threshold": PROMOTION_THRESHOLD,
            "champion_source": champion_dir,
            "promoted": promoted
        })
        mlflow.set_tags({
            "stage": "evaluation",
            "decision": "promoted" if promoted else "kept_champion"
        })

    if promoted:
        promote_challenger()

    # Save result for GitHub Actions to read
    result = {
        "promoted": promoted,
        "challenger_f1": challenger_metrics["f1"],
        "champion_f1": champion_metrics["f1"],
        "f1_delta": f1_delta
    }
    with open("mlops/eval_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to mlops/eval_result.json")

    return result

if __name__ == "__main__":
    result = run_evaluation()
    print(f"\nFinal result: {result}")