import joblib
joblib.dump(best_model, 'models/amazon_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

model = joblib.load('models/amazon_model.pkl')
scaler = joblib.load('models/scaler.pkl')
analyzer = SentimentIntensityAnalyzer()

def clean_currency(x):
    """Nettoie le prix comme dans la partie 2.1 du notebook"""
    if isinstance(x, str):
        x = x.replace('₹', '').replace(',', '').strip()
        return float(x)
    return float(x)

def text_cleaning(text):
    """Nettoie le texte comme dans ta pipeline NLP"""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

st.title("🚀 Amazon Product Rating Predictor")
st.markdown("Enter product details below to predict the customer rating.")

price_input = st.text_input("Product Price (e.g., ₹1,299)", "₹500")
review_title = st.text_input("Review Title", "Great product!")
review_content = st.text_area("Review Content", "I really enjoyed using this, it works perfectly.")
rating_count = st.number_input("Number of total ratings for this product", min_value=1, value=100)

if st.button("Predict Rating"):

    price_numeric = clean_currency(price_input)
    log_price = np.log1p(price_numeric)

    full_text = text_cleaning(review_title + " " + review_content)
    sentiment_score = analyzer.polarity_scores(full_text)['compound']
    
    review_len = len(full_text)
    word_count = len(full_text.split())

    features = pd.DataFrame([[rating_count, sentiment_score, review_len, word_count, log_price]], 
                            columns=['rating_count', 'sentiment_score', 'review_len', 'word_count', 'log_price'])

    features_scaled = scaler.transform(features)
  
    prediction = model.predict(features_scaled)

    st.success(f"### Predicted Rating: {prediction[0]:.2f} / 5 ⭐")
