# predict.py
import argparse
import joblib
import numpy as np
import tensorflow as tf
from utils_text_cleaning import clean_text
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_ml(model_path, vectorizer_path, text):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    clean = clean_text(text)
    X = vectorizer.transform([clean])
    pred = model.predict(X)[0]
    return "Positive" if pred == 1 else "Negative"

def predict_rnn(model_path, tokenizer_path, text, max_len=40):
    model = tf.keras.models.load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    pred = model.predict(pad)[0][0]
    return "Positive" if pred >= 0.5 else "Negative"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model_type", choices=["ml", "rnn"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vectorizer", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max_len", type=int, default=40)
    args = parser.parse_args()

    if args.model_type == "ml":
        result = predict_ml(args.model, args.vectorizer, args.text)
    else:
        result = predict_rnn(args.model, args.tokenizer, args.text, args.max_len)
    print(f"Prediction: {result}")
