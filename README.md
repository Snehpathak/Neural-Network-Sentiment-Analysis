# Neural-Network-Sentiment-Analysis
This project implements a sentiment analysis pipeline to classify text as positive or negative using both traditional machine learning (ML) models and a recurrent neural network (RNN). It includes data preprocessing, model training, evaluation, and a Streamlit-based user interface to compare model performance and make predictions.

Project Structure





data_prep.py: Preprocesses the dataset, cleans text, and generates TF-IDF features for ML models.



utils_text_cleaning.py: Contains utility functions for text cleaning (removing URLs, punctuation, stopwords, and lemmatization).



train_ml.py: Trains and evaluates ML models (Logistic Regression, SVM, Naive Bayes, Random Forest) using TF-IDF features.



train_rnn.py: Trains and evaluates an RNN model (LSTM-based) using tokenized and padded text sequences.



predict.py: Provides functions to make predictions using trained ML or RNN models.



app.py: A Streamlit application to display validation metrics (accuracy and F1-score) for all models and allow users to input text for sentiment predictions.



Setup and Usage





Clone the Repository:

git clone <repository-url>
cd <repository-directory>



Prepare the Dataset:





Place your dataset (e.g., twitter.csv) in a data directory or update the DATA_PATH in app.py.



Preprocess Data:

python data_prep.py --input_csv data/training.1600000.processed.noemoticon.csv --out_dir features





Optionally, use --limit <number> to process a subset of the data (e.g., --limit 10000).



Train ML Models:

python train_ml.py --features_dir features --out_dir ml_models





This trains Logistic Regression, SVM, Naive Bayes, and Random Forest models and saves them to ml_models.



Train RNN Model:

python train_rnn.py --input_csv data/twitter.csv --out_dir rnn_models





Optionally, use --limit <number> for faster training on a subset.



Run the Streamlit App:





Update the paths in app.py (DATA_PATH, FEATURES_DIR, ML_OUT_DIR, RNN_OUT_DIR) to match your directory structure.



Run the app:

streamlit run app.py



Open the provided URL (e.g., http://localhost:8501) in a browser.



Using the Streamlit App:





Validation Metrics: View a table showing accuracy and F1-score for all models (Logistic Regression, SVM, Naive Bayes, Random Forest, RNN) on the test set.



Predict Sentiment: Enter text in the text area and click "Predict" to see sentiment predictions from all models.
