"""
A simple Union app using Streamlit to serve a BERT model with Streamlit.
"""

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from union_runtime import get_input

# Load the model artifact downloaded by Union.
model_path = get_input("bert_model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Title and description
st.title("Sentiment Analyzer")
st.write("Enter text to predict the sentiment.")

# Input text for sentiment analysis
user_input = st.text_area("Enter your text:", height=400, key="text_input")

if st.button("Analyze"):
    try:
        # Tokenize and move inputs to same device as model
        inputs = tokenizer(
            user_input, return_tensors="pt", truncation=True, padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        labels = ["NEGATIVE", "POSITIVE"]  # Adjust if your model has different classes

        sentiment = labels[predictions.item()]
        score = probabilities[0][predictions.item()].item()

        if sentiment == "NEGATIVE":
            st.error(f"Predicted sentiment: {sentiment} (Confidence: {score:.2f})")
        else:
            st.success(f"Predicted sentiment: {sentiment} (Confidence: {score:.2f})")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Deploy using:
# union deploy apps app.py bert-sentiment-analysis
