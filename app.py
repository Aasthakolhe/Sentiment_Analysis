from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import numpy as np
import os
import sys

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.preprocessing import preprocess_text

app = FastAPI(title="Context & Confidence-Aware Sentiment Dashboard")
templates = Jinja2Templates(directory="templates")

# Load model artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'sentiment_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

class FeedbackRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def serve_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict_sentiment(request: FeedbackRequest):
    raw_text = request.text
    cleaned_text = preprocess_text(raw_text)
    
    probabilities = model.predict_proba([cleaned_text])[0]
    classes = model.classes_
    max_idx = int(np.argmax(probabilities))
    
    predicted_sentiment = str(classes[max_idx])
    confidence = float(probabilities[max_idx])
    
    # Uncertainty Threshold (Below 65% triggers warning)
    CONFIDENCE_THRESHOLD = 0.65
    is_uncertain = bool(confidence < CONFIDENCE_THRESHOLD)
    
    return {
        "original_text": raw_text,
        "preprocessed_text": cleaned_text,
        "predicted_sentiment": predicted_sentiment,
        "confidence_score": f"{round(confidence * 100, 2)}%",
        "is_uncertain": is_uncertain,
        "uncertainty_warning": (
            "⚠️ Warning: Low prediction confidence. Sentence may contain ambiguous sentiment, "
            "sarcasm, or context-dependent Gen Z slang."
            if is_uncertain else "High confidence prediction."
        )
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)