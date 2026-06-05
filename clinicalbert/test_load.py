from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Test it
text = "Patient shows signs of memory impairment and word-finding difficulty."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
print(outputs.logits)