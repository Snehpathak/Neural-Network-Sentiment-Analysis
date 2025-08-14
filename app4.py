import streamlit as st
import joblib
import tensorflow as tf
import pandas as pd
from utils_text_cleaning import clean_text
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

# Assume paths for model, feature, and metrics artifacts
FEATURES_DIR = r"C:\Users\lenovo\Documents\DSMM T3\Neural Networks and Deep Learning 01\Project\feature"  # From data_prep.py out_dir
ML_OUT_DIR = r"C:\Users\lenovo\Documents\DSMM T3\Neural Networks and Deep Learning 01\Project\models"  # From train_ml.py out_dir
RNN_OUT_DIR = r"C:\Users\lenovo\Documents\DSMM T3\Neural Networks and Deep Learning 01\Project"  # From train_rnn.py out_dir
METRICS_PATH = r"C:\Users\lenovo\Documents\DSMM T3\Neural Networks and Deep Learning 01\Project\metrics.json"
MAX_LEN = 200

# Load models and vectorizer/tokenizer
vectorizer = joblib.load(f"{FEATURES_DIR}/tfidf_vectorizer.joblib")
models = {
    "Logistic Regression": joblib.load(f"{ML_OUT_DIR}/logreg.joblib"),
    "SVM": joblib.load(f"{ML_OUT_DIR}/svm.joblib"),
    "Naive Bayes": joblib.load(f"{ML_OUT_DIR}/naive_bayes.joblib"),
    "Random Forest": joblib.load(f"{ML_OUT_DIR}/random_forest.joblib"),
    "LSTM": tf.keras.models.load_model(f"{RNN_OUT_DIR}/rnn_model.keras")
}
tokenizer = joblib.load(f"{RNN_OUT_DIR}/tokenizer.joblib")

# Load metrics from JSON
with open(METRICS_PATH, 'r') as f:
    metrics = json.load(f)

# Streamlit UI
st.title("Sentiment Analysis")

# Model selection
model_name = st.selectbox("Choose a model:", list(models.keys()))

# Text input
text = st.text_input("Enter text for sentiment prediction:")

if st.button("Predict"):
    if text:
        # Display metrics for selected model
        st.header(f"{model_name} Performance Metrics")
        metrics_df = pd.DataFrame(metrics[model_name], index=[model_name]).T
        st.table(metrics_df)

        # Prediction
        clean = clean_text(text)
        if model_name == "LSTM":
            seq = tokenizer.texts_to_sequences([clean])
            pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
            pred = models[model_name].predict(pad)[0][0]
            label = "Positive" if pred >= 0.5 else "Negative"
        else:
            X = vectorizer.transform([clean])
            pred = models[model_name].predict(X)[0]
            label = "Positive" if pred == 1 else "Negative"
        
        st.write(f"Sentiment Prediction: {label}")
    else:
        st.write("Please enter some text.")