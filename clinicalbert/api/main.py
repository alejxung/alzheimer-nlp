from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

#=== Config =============================================#
CHECKPOINT_DIR = "clinicalbert/checkpoints/best_model"
MAX_LEN = 512

#=== Device =============================================#
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()

#=== Request / Response schemas =========================#
class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str         # "dementia" or "control"
    confidence: float       # probability of the predicted class
    probabilities: dict     # full breakdown: {"control": 0.13, "dementia": 0.87}

#=== App + model loading ================================#
app = FastAPI(title="Alzheimer's NLP Classifier", version="1.0")

print(f"Loading model from {CHECKPOINT_DIR} on {device}...")

peft_config = PeftConfig.from_pretrained(CHECKPOINT_DIR)
base_model = AutoModelForSequenceClassification.from_pretrained(
    peft_config.base_model_name_or_path, num_labels=2
)
model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

model = model.to(device)
model.eval()

print("Model loaded. API ready.")

LABEL_MAP = {0: "control", 1: "dementia"}

#=== Routes ==============================================#
@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    inputs = tokenizer(
        request.text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu()

    pred_idx = int(torch.argmax(probs).item())

    return PredictResponse(
        prediction=LABEL_MAP[pred_idx],
        confidence=float(probs[pred_idx]),
        probabilities={
            "control": float(probs[0]),
            "dementia": float(probs[1])
        }
    )