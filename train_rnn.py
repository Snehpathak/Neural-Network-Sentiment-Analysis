# train_rnn.py
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import joblib
from utils_text_cleaning import clean_text

def main(input_csv, out_dir, limit=None, max_len=200):
    print("[INFO] Loading dataset...")
    data = pd.read_csv(input_csv, encoding='latin', names=['polarity','id','date','query','user','text'])
    if limit:
        data = data.sample(frac=1).head(limit)
    data['polarity'] = data['polarity'].replace(4, 1)
    data['clean_text'] = data['text'].apply(clean_text)

    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(data['clean_text'])
    sequences = tokenizer.texts_to_sequences(data['clean_text'])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    X_train, X_test, y_train, y_test = train_test_split(padded, data['polarity'], test_size=0.2, random_state=42)

    model = Sequential([
        Embedding(5000, 128, input_length=max_len),
        LSTM(32, return_sequences=False),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("[INFO] Training RNN...")
    model.fit(X_train, y_train, epochs=8, validation_data=(X_test, y_test), batch_size=64)

    model.save(f"{out_dir}/rnn_best1.keras")
    joblib.dump(tokenizer, f"{out_dir}/tokenizer.joblib")
    print("[INFO] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.input_csv, args.out_dir, args.limit)
