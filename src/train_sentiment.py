import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from sklearn.metrics import classification_report

# Download French stopwords (Only need to run once)
#nltk.download('stopwords')


french_stopwords = stopwords.words('french')
df = pd.read_csv("../data/cleaned/review_dataset.csv")
X_train, X_test, y_train, y_test = train_test_split(df['Review Text'], df['sentiment'], test_size=0.2, random_state=42)
# TF-IDF Vectorizer with French Stopwords
tfidf_vectorizer = TfidfVectorizer(stop_words=french_stopwords, max_features=5000)
# Transform text
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
# Evaluate model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Sentiment Model Accuracy: {accuracy:.2f}")
# Save Model & Vectorizer
joblib.dump(model, "../models/sentiment_model_fr.pkl")
joblib.dump(tfidf_vectorizer, "../models/tfidf_vectorizer_fr.pkl")
report = classification_report(y_test, y_pred)
with open("../reports/classification_report.txt", "w") as f:
    f.write(report)
cm = confusion_matrix(y_train, y_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Négatif", "Positif"], yticklabels=["Négatif", "Positif"])
plt.savefig('../reports/confusion_matrix.png')

