import re
import numpy as np
from scipy.special import softmax
import torch

# -------- Clean Tweet Text --------
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)            # Remove URLs
    text = re.sub(r"<.*?>", "", text)              # Remove HTML tags
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)     # Remove punctuation
    return text.strip().lower()

# -------- Get RoBERTa Scores --------
def get_roberta_scores(text: str, tokenizer, model) -> dict:
    encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**encoded)
        logits = output.logits.detach().cpu().numpy()
        probs = softmax(logits, axis=1)[0]  # shape: (3,)
        return {
            "roberta_pos": float(probs[2]),
            "roberta_neu": float(probs[1]),
            "roberta_neg": float(probs[0])
        }

# -------- Extract Features for ML Model --------
def extract_features(original_text: str, cleaned_text: str, sia, tokenizer, model) -> list:
    vader_scores = sia.polarity_scores(cleaned_text)
    roberta_scores = get_roberta_scores(cleaned_text, tokenizer, model)

    features = [
        vader_scores['compound'],
        roberta_scores['roberta_pos'],
        roberta_scores['roberta_neu'],
        roberta_scores['roberta_neg'],
        len(cleaned_text),
        len(cleaned_text.split()),
        int('!' in original_text),
        int('?' in original_text)
    ]
    
    return features