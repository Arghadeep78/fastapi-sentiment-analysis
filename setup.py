import os
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_nltk_resources():
    print("🔽 Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('vader_lexicon')
    print("✓ NLTK resources downloaded.")

def create_project_folders():
    print("📁 Ensuring project folders exist...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    print("✓ Project folders are ready.")

def download_roberta_model():
    print("🔽 Downloading RoBERTa model and tokenizer (cached by HuggingFace)...")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    print("✓ RoBERTa model and tokenizer downloaded and cached.")

if __name__ == "__main__":
    print("\n🚀 Running setup...\n")
    download_nltk_resources()
    create_project_folders()

    # Optional: uncomment if you want to pre-download RoBERTa (uses local Hugging Face cache)
    # download_roberta_model()

    print("\n✅ Setup complete! You're ready to run the app.\n")
