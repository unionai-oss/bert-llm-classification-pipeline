"""A simple Union app using Streamlit to serve an bert model with streamlit."""

import streamlit as st
from union_runtime import get_input
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# Load the model artifact downloaded by Union.
model_path = get_input("bert_model")
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Title and description
st.title("Sentiment Analyzer")
st.write("Enter text to predict the sentiment.")

# Input text for sentiment analysis
user_input = st.text_area("Enter your text:")

if st.button("Analyze"):
    try:
        # Tokenize and predict
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
        labels = ["NEGATIVE", "POSITIVE"]  # Adjust according to your model's labels

        sentiment = labels[predictions.item()]
        if sentiment == "NEGATIVE":
            st.error(f"Predicted sentiment: {sentiment}")
        else:
            st.success(f"Predicted sentiment: {sentiment}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# union deploy apps app.py bert-sentiment-analysis