# train_ml.py
import argparse
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main(features_dir, out_dir):
    X_train, X_test, y_train, y_test = joblib.load(f"{features_dir}/tfidf_data.joblib")

    models = {
        "logreg": LogisticRegression(),
        "svm": LinearSVC(),
        "naive_bayes": BernoulliNB(),
        "random_forest": RandomForestClassifier(n_estimators=20,max_depth=50)
    }

    for name, model in models.items():
        print(f"[INFO] Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"[RESULT] {name} performance:\n", classification_report(y_test, preds))
        joblib.dump(model, f"{out_dir}/{name}.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args.features_dir, args.out_dir)


metrics = {
    "Naive Bayes": {
        "accuracy": accuracy_score(y_val, nb_pred),
        "precision": precision_score(y_val, nb_pred, average="weighted"),
        "recall": recall_score(y_val, nb_pred, average="weighted"),
        "f1": f1_score(y_val, nb_pred, average="weighted")
    },
    "SVM": {
        "accuracy": accuracy_score(y_val, svm_pred),
        "precision": precision_score(y_val, svm_pred, average="weighted"),
        "recall": recall_score(y_val, svm_pred, average="weighted"),
        "f1": f1_score(y_val, svm_pred, average="weighted")
    }
}

# Save metrics
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f)    
