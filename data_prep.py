# data_prep.py
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
from utils_text_cleaning import clean_text

def main(input_csv, out_dir, limit=None):
    print("[INFO] Loading dataset...")
    data = pd.read_csv(input_csv, encoding='latin', names=['polarity','id','date','query','user','text'])
    if limit:
        data = data.sample(frac=1).head(limit)

    # Convert polarity from 0/4 to 0/1
    data['polarity'] = data['polarity'].replace(4, 1)

    print("[INFO] Cleaning text...")
    data['clean_text'] = data['text'].apply(clean_text)

    print("[INFO] Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data['clean_text'])
    y = data['polarity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Saving artifacts...")
    joblib.dump(vectorizer, f"{out_dir}/tfidf_vectorizer.joblib")
    joblib.dump((X_train, X_test, y_train, y_test), f"{out_dir}/tfidf_data.joblib")
    print("[INFO] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.input_csv, args.out_dir, args.limit)
