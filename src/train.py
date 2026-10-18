import pandas as pd
import numpy as np
import pickle
import os
import sys
import time

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def train_sentiment_model():
    # 1. Load dataset
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reviews.csv')
    print(f"Loading dataset from {data_path}...")
    
    df = pd.read_csv(data_path)
    valid_sentiments = ['Positive', 'Negative', 'Neutral', 'Mixed']
    df = df[df['sentiment'].isin(valid_sentiments)].dropna(subset=['text', 'sentiment']).copy()
    
    print(f"Total rows to train on: {len(df)}")
    
    # 2. Apply context-aware preprocessing
    print("Applying context-aware preprocessing (emojis & Gen Z slang)...")
    start_time = time.time()
    df['clean_text'] = df['text'].apply(preprocess_text)
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds.")
    
    # 3. Train Sentiment Model
    print("Training Sentiment Classifier...")
    model_pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 3), max_features=35000),
        LogisticRegression(max_iter=1000)
    )
    
    model_pipeline.fit(df['clean_text'], df['sentiment'])
    print("Model training complete!")
    
    # 4. Save model artifact
    os.makedirs('models', exist_ok=True)
    model_save_path = 'models/sentiment_model.pkl'
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_pipeline, f)
        
    print(f"[SUCCESS] Model successfully saved to {model_save_path}!")

if __name__ == "__main__":
    train_sentiment_model()