import joblib

# Load
sentiment_model = joblib.load("../models/sentiment_model_fr.pkl")
tfidf_vectorizer = joblib.load("../models/tfidf_vectorizer_fr.pkl")


def predict_sentiment(sample):
     #transfromer texte d'abord
    text_tfidf = tfidf_vectorizer.transform([sample])  #must be a list

    # 1 for Positive, 0 for Negative
    sentiment_pred = sentiment_model.predict(text_tfidf)[0]
    sentiment_label = "Positif" if sentiment_pred == 1 else "Négatif"
    return sentiment_label


text = "Excellent. Très bonne banque !Remarque les esprits chagrins qui mettent 1 étoile à wafabank  doivent savoir que wafabank meilleure banque au Maroc et plus n'accueille pas n'importe qui avec le sourire pour des personnes peu claires avec des rentrées d'argent hypothétiques et des projets fumeux."
predicted_sentiment = predict_sentiment(text)
print(f"Review: {text}")
print(f"Predicted Sentiment: {predicted_sentiment}")

