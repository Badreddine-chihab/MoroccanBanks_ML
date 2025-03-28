import joblib

# Load
model = joblib.load("../models/sentiment_model_fr.pkl")

# Load the pre-trained TF-IDF vectorizer
vectorizer = joblib.load("../models/tfidf_vectorizer_fr.pkl")
def predict_star_rating(review_text):
    # Transform review text into numerical features
    review_tfidf = vectorizer.transform([review_text])


    pred_rating = model.predict(review_tfidf)

    rounded_rating = round(pred_rating[0] * 2) / 2 #round rating pour avoir un rating plus logique

    return rounded_rating

# message test
new_review = "service mediocre"
predicted_rating = predict_star_rating(new_review)

print(f"Predicted Star Rating: {predicted_rating} star")
