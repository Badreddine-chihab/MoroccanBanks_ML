# MoroccanBanks_ML
This project will display our python and ML skills on a real moroccan dataset extracted from data.gov.ma

Sentiment Analysis and Star Rating Prediction for Moroccan Bank Reviews

Project Overview

This project aims to analyze and predict sentiment and star ratings from Moroccan bank reviews collected from Google Maps. The dataset has been cleaned, analyzed, and used to build a machine learning model for sentiment analysis. Additionally, a PyQt5-based GUI was developed for user-friendly interaction with the sentiment prediction model.

Project Workflow

1. Dataset Cleaning and Preparation

Initial Data Cleaning: Removed unnecessary columns, handled missing values, and corrected city names.

Preprocessing: Tokenization, stopword removal (using French stopwords), and TF-IDF vectorization.

2. Exploratory Data Analysis (EDA)

Data Visualization: Used Seaborn and Matplotlib to analyze sentiment distributions and review patterns.

Correlation Analysis: Explored relationships between review text and star ratings.

3. Sentiment Analysis Model

Training Model: Implemented Logistic Regression with TF-IDF vectorization.

Model Evaluation: Assessed performance using accuracy score, classification report, and confusion matrix.

Saving Model & Vectorizer: Exported trained models using Joblib.

4. Predicting Star Ratings

Sentiment-Based Prediction: Estimated star ratings based on sentiment analysis results.

Integration with GUI: Developed a PyQt5-based application for real-time predictions.

5. PyQt5 GUI Implementation

Class: ********************SentimentAnalyser: Created a Python class for handling sentiment predictions.

User Interface: Designed an interactive GUI allowing users to enter reviews and receive sentiment predictions instantly.

Model Integration: Loaded the trained model and vectorizer to process user inputs.

Results

Achieved a sentiment classification accuracy of X.XX% (to be updated with the actual result).

Successfully visualized sentiment trends and relationships in Moroccan bank reviews.

Deployed a functional PyQt5-based GUI for real-time sentiment prediction.

How to Run the Project

Requirements

Install the necessary dependencies:

pip install pandas joblib scikit-learn nltk seaborn matplotlib PyQt5

Running the Sentiment Analyzer

python sentiment_gui.py

This will launch the GUI, allowing users to input reviews and receive sentiment predictions.

Future Enhancements

Improve model accuracy using deep learning techniques (e.g., LSTMs or Transformers).

Expand dataset to include more Moroccan bank reviews.

Deploy the model as a web application for wider accessibility.

Author

Developped by me and Amine Chakri for  "End-of-Term Project"
