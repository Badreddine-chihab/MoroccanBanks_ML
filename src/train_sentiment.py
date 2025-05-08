import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report

# Load data
df = pd.read_csv("../data/cleaned/review_dataset.csv")


# Split data
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['Review Text'])
X_test = vectorizer.transform(test_df['Review Text'])

# Train model
model = GradientBoostingClassifier(n_estimators=99, learning_rate=0.1, max_depth=3)
model.fit(X_train, train_df['sentiment'])

# Evaluate
preds = model.predict(X_test)
print(classification_report(test_df['sentiment'], preds))
print("Confusion Matrix:\n", confusion_matrix(test_df['sentiment'], preds))


joblib.dump(vectorizer, '../models/vectorizer.pkl')
joblib.dump(model, '../models/model_sentiment.pkl')