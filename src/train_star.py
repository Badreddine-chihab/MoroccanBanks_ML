import pandas as pd
import joblib
import os
import json
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Download French stop words
#nltk.download("stopwords")
french_stopwords = stopwords.words("french")
df = pd.read_csv("../data/cleaned/updated_dataset.csv")  # Make sure this file exists
X = df["Review Text"]
y = df["Stars"]
# load
vectorizer_path = "../models/tfidf_vectorizer_fr.pkl"
print("Loading pre-trained TF-IDF vectorizer...")
vectorizer = joblib.load(vectorizer_path)
X_tfidf = vectorizer.transform(X)
#split data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=43)
# Train Gradient Boosting
model = GradientBoostingRegressor(n_estimators=500,learning_rate=0.1,random_state = 101)
model.fit(X_train, y_train)
# Predict ratings on test data
y_pred = model.predict(X_test)
# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
# Save trained model
joblib.dump(model, "../models/sentiment_model_fr.pkl")
# Save evaluation results
results = {"Mean Squared Error": mse}
with open("../reports/evaluation_results.json", "w") as f:
    json.dump(results, f)

print("Model saved in 'models/' and results saved in 'reports/'")
