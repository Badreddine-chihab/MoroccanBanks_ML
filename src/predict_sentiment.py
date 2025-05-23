import joblib

# Load
sentiment_model = joblib.load("../models/model_sentiment.pkl")
tfidf_vectorizer = joblib.load("../models/vectorizer.pkl")


def predict_sentiment(sample):
     #transfromer texte d'abord
    text_tfidf = tfidf_vectorizer.transform([sample])  #must be a list

    # 1 for Positive, 0 for Negative
    sentiment_pred = sentiment_model.predict(text_tfidf)[0]
    sentiment_label = "Positif" if sentiment_pred == 1 else "Négatif"
    return sentiment_label

print(predict_sentiment("gentils"))

