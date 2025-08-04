# 🧠 FastAPI Sentiment Analysis Web App

This project is a full-stack **Sentiment Analysis** app using:

- 🤖 RoBERTa Transformer (Hugging Face)
- 🌲 Random Forest classifier
- 💬 VADER Sentiment Analyzer
- ⚡ FastAPI backend
- 🌐 HTML/CSS frontend

## 📊 Model Training (`model.ipynb`)

The `model.ipynb` notebook trains a Random Forest classifier using combined features from:

- **VADER** sentiment scores  
- **RoBERTa** sentiment probabilities  
- Text-based features: `text_len`, `num_words`, `has_exclam`, `has_question`, etc.

The resulting model achieved:

- **F1 Score**: `0.75`  
- **Accuracy**: `75%`  


---

## 📁 Project Structure

```bash
.
├── __pycache__/
├── models/
│   ├── roberta_model/           # RoBERTa tokenizer & model files
│   ├── scaler.pkl               # StandardScaler used with Random Forest
│   └── sentiment_model.pkl      # Trained Random Forest model
├── static/
│   └── style.css                # CSS for frontend
├── templates/
│   └── index.html               # HTML template rendered by FastAPI
├── screenshots/                 # Screenshots of FastAPI app
│   └── screenshot_01.jpeg
│   └── screenshot_02.jpeg
├── train.csv                    # Sample training data
├── model.ipynb                  # Jupyter Notebook for model training
├── app.py                       # Main FastAPI app
├── utils.py                     # Helper functions
├── setup.py                     # Optional setup file
└── requirements.txt             # Python dependencies
└── README.md                    # User Guide

```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
uvicorn app:app --reload
```

Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 💡 Features

- Predict sentiment using **RoBERTa**
- Compare results from **Random Forest** and **VADER**
- Web UI built with **FastAPI**, rendered via `index.html`
- Lightweight frontend with `style.css`

---

## ✅ Status

✔️ All models and assets are already uploaded to the repo — no external downloads needed.

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [scikit-learn](https://scikit-learn.org/)
- [NLTK](https://www.nltk.org/)

- 
## 📬 Contact

Created by **[Arghadeep Ghosh]**  
🔗 [LinkedIn](https://www.linkedin.com/in/arghadeep-ghosh-895b27287/)  
📫 [Email](arghadeepghosh17@gmail.com)

---

> ⭐ Star this repo if you like the project or find it helpful!
