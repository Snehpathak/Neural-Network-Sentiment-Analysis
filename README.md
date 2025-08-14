# Neural-Network-Sentiment-Analysis
This project implements a sentiment analysis pipeline to classify text as positive or negative using both traditional machine learning (ML) models and a recurrent neural network (RNN). It includes data preprocessing, model training, evaluation, and a Streamlit-based user interface to compare model performance and make predictions.

Project Structure





data_prep.py: Preprocesses the dataset, cleans text, and generates TF-IDF features for ML models.



utils_text_cleaning.py: Contains utility functions for text cleaning (removing URLs, punctuation, stopwords, and lemmatization).



train_ml.py: Trains and evaluates ML models (Logistic Regression, SVM, Naive Bayes, Random Forest) using TF-IDF features.



train_rnn.py: Trains and evaluates an RNN model (LSTM-based) using tokenized and padded text sequences.



predict.py: Provides functions to make predictions using trained ML or RNN models.



app.py: A Streamlit application to display validation metrics (accuracy and F1-score) for all models and allow users to input text for sentiment predictions.
