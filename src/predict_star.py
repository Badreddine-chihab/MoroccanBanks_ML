import joblib
import numpy as np
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self, model_path: str, vectorizer_path: str):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

        # Define thresholds for mapping rating to sentiment
        self.thresholds = {
            'positive': 3.5,
            'neutral': 2.5,
            'negative': 0.0
        }

    def transform_text(self, text: str):
        return self.vectorizer.transform([text])

    def predict_rating(self, text: str) -> float:

        try:
            tfidf = self.transform_text(text)
            rating = self.model.predict(tfidf)[0]
            return float(rating)
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return 3.0  # Default neutral rating

    def predict_sentiment(self, rating: float) -> str:

        if rating >= self.thresholds['positive']:
            return "positive"
        elif rating >= self.thresholds['neutral']:
            return "neutral"
        else:
            return "negative"

    def calculate_confidence(self, text: str, predicted_rating: float) -> float:
        """
        Calculate confidence score from predicted class probability.
        """
        try:
            tfidf = self.transform_text(text)
            probas = self.model.predict_proba(tfidf)[0]
            class_idx = int(predicted_rating) - 1  # e.g., rating 4 → index 3
            confidence = probas[class_idx] if 0 <= class_idx < len(probas) else max(probas)
            return round(float(confidence), 2)
        except Exception as e:
            logger.warning(f"Could not calculate confidence: {str(e)}")
            return 0.5  # default confidence

    def analyze_review(self, text: str) -> Dict[str, object]:
        """
        Analyze a review and return rating, sentiment, and confidence.
        """
        rating = self.predict_rating(text)
        sentiment = self.predict_sentiment(rating)
        confidence = self.calculate_confidence(text, rating)

        return {
            "rating": rating,
            "sentiment": sentiment,
            "confidence": confidence
        }


if __name__ == "__main__":
    analyzer = SentimentAnalyzer("../models/model_star.pkl", "../models/vectorizer.pkl")

    test_reviews = [
        "Service excellent, je suis très satisfait!",
        "Très déçu par la qualité, je ne recommande pas",
        "Bon service",
        "Médiocre, ne correspond pas à la description",
        "Excellent. Très bonne banque !"
    ]

    for review in test_reviews:
        result = analyzer.analyze_review(review)
        print(f"Review: {review[:60]}...")
        print(f"Rating: {result['rating']} stars")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']}")
        print("-" * 50)

