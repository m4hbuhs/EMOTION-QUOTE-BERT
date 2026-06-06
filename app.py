import streamlit as st
import torch
import pickle
import pandas as pd
import random

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
model = AutoModelForSequenceClassification.from_pretrained("bert_emotion_model")
tokenizer = AutoTokenizer.from_pretrained("bert_emotion_model")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load quotes
quotes_df = pd.read_csv("quotes.csv",on_bad_lines='skip')

# Prediction
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]

# Recommendation
def recommend_quote(text):
    emotion = predict_emotion(text)
    filtered = quotes_df[quotes_df['emotion'] == emotion]

    if len(filtered) > 0:
        quote = random.choice(filtered['quote'].tolist())
    else:
        quote = "Stay positive!"

    return emotion, quote

# UI
st.title("🧠 Emotion AI")

text = st.text_area("Enter your feelings:")

if st.button("Predict"):
    if text:
        emotion, quote = recommend_quote(text)

        st.success(f"Emotion: {emotion}")
        st.info(f"Quote: {quote}")
    else:
        st.warning("Enter something!")