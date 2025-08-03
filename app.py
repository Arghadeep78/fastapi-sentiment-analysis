from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer

from utils import clean_text, extract_features, get_roberta_scores

# Initialize FastAPI app
app = FastAPI()

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Load models from local directory
tokenizer = AutoTokenizer.from_pretrained("models/roberta_model")
roberta_model = AutoModelForSequenceClassification.from_pretrained("models/roberta_model")
vader = SentimentIntensityAnalyzer()
scaler = joblib.load("models/scaler.pkl")
classifier = joblib.load("models/sentiment_model.pkl")

# Define input schema for POST /predict
class TextInput(BaseModel):
    text: str

# === ROUTES ===

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction route
@app.post("/predict")
async def predict_sentiment(data: TextInput):
    tweet = data.text
    cleaned = clean_text(tweet)
    
    # Get scores from both models
    vader_scores = vader.polarity_scores(cleaned)
    roberta_scores = get_roberta_scores(cleaned, tokenizer, roberta_model)
    
    # Extract features and predict sentiment
    features = extract_features(tweet, cleaned, vader, tokenizer, roberta_model)
    scaled = scaler.transform([features])
    prediction = classifier.predict(scaled)[0]

    # Return full JSON response
    return JSONResponse(content={
        "predicted_sentiment": prediction,
        "vader": vader_scores,
        "roberta": roberta_scores
    })