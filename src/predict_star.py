import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json
from pathlib import Path
import logging
from typing import Tuple, Dict, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    A comprehensive sentiment analysis system for review rating prediction
    """

    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        """
        Initialize the sentiment analyzer with model and vectorizer paths

        Args:
            model_path: Path to the trained model file
            vectorizer_path: Path to the TF-IDF vectorizer file
        """
        self.model = None
        self.vectorizer = None
        self.label_mapping = None
        self.thresholds = {
            'positive': 4.0,
            'neutral': 3.0,
            'negative': 2.0
        }

        # Load configuration
        self._load_config()

        # Load model and vectorizer
        if model_path and vectorizer_path:
            self.load_models(model_path, vectorizer_path)
        elif hasattr(self, '../models/star_model_fr.pkl') and hasattr(self, '../models/tfidf_vectorizer.pkl'):
            self.load_models(self.default_model_path, self.default_vectorizer_path)

    def _load_config(self):
        """Load configuration from JSON file if available"""
        config_path = Path(__file__).parent / "config.json"
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.default_model_path = config.get('model_path')
                    self.default_vectorizer_path = config.get('vectorizer_path')
                    self.thresholds = config.get('thresholds', self.thresholds)
                    self.label_mapping = config.get('label_mapping')
                    logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load configuration: {str(e)}")

    def load_models(self, model_path: str, vectorizer_path: str) -> bool:
        """
        Load the trained model and vectorizer from disk

        Args:
            model_path: Path to the trained model file
            vectorizer_path: Path to the TF-IDF vectorizer file

        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            logger.info("Model and vectorizer loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before vectorization (can be extended for language-specific processing)

        Args:
            text: Input text to preprocess

        Returns:
            str: Preprocessed text
        """
        # Basic preprocessing - extend this for French-specific processing
        text = text.lower().strip()
        return text

    def transform_text(self, text: str):
        """
        Transform text into numerical features using the loaded vectorizer

        Args:
            text: Input text to transform

        Returns:
            Sparse matrix: Transformed text features
        """
        if not self.vectorizer:
            raise ValueError("Vectorizer not loaded. Please load the vectorizer first.")

        preprocessed_text = self.preprocess_text(text)
        return self.vectorizer.transform([preprocessed_text])

    def predict_star_rating(self, review_text: str) -> float:
        """
        Predict star rating for a given review text

        Args:
            review_text: Text of the review to analyze

        Returns:
            float: Predicted star rating (0.5 to 5.0 in 0.5 increments)
        """
        if not self.model or not self.vectorizer:
            raise ValueError("Model or vectorizer not loaded. Please load models first.")

        try:
            # Transform review text into numerical features
            review_tfidf = self.transform_text(review_text)

            # Make prediction
            pred_rating = self.model.predict(review_tfidf)[0]

            # Round to nearest 0.5 (1.0, 1.5, 2.0, etc.)
            rounded_rating = round(pred_rating * 2) / 2
            if rounded_rating > 5:
                rounded_rating = 5
            elif rounded_rating < 1:
                rounded_rating = 1

            return rounded_rating
        except Exception as e:
            logger.error(f"Error predicting rating: {str(e)}")
            raise

    def analyze_sentiment(self, review_text: str) -> Dict[str, Union[str, float]]:
        """
        Perform comprehensive sentiment analysis on review text

        Args:
            review_text: Text of the review to analyze

        Returns:
            dict: Dictionary containing analysis results including:
                - rating: Predicted star rating
                - sentiment: Sentiment category (positive/neutral/negative)
                - confidence: Prediction confidence score
        """
        rating = self.predict_star_rating(review_text)

        # Determine sentiment category based on thresholds
        if rating >= self.thresholds['positive']:
            sentiment = "positive"
        elif rating >= self.thresholds['neutral']:
            sentiment = "neutral"
        else:
            sentiment = "negative"

        # Calculate confidence (placeholder - implement your own confidence measure)
        confidence = self._calculate_confidence(review_text, rating)

        return {
            'rating': rating,
            'sentiment': sentiment,
            'confidence': confidence,
            'text': review_text[:100] + "..." if len(review_text) > 100 else review_text
        }

    def _calculate_confidence(self, text: str, rating: float) -> float:
        """
        Calculate prediction confidence (placeholder implementation)

        Args:
            text: Review text
            rating: Predicted rating

        Returns:
            float: Confidence score between 0 and 1
        """
        # Placeholder - implement a proper confidence measure
        # This could be based on prediction probabilities, text length, etc.
        text_length = len(text)
        base_confidence = 0.7  # Base confidence
        length_factor = min(1.0, text_length / 50)  # More confidence for longer texts

        return min(1.0, base_confidence * length_factor)

    def get_sentiment_stats(self, reviews: list) -> Dict[str, float]:
        """
        Calculate statistics for a batch of reviews

        Args:
            reviews: List of review texts

        Returns:
            dict: Dictionary containing sentiment statistics including:
                - avg_rating: Average star rating
                - positive_pct: Percentage of positive reviews
                - negative_pct: Percentage of negative reviews
                - neutral_pct: Percentage of neutral reviews
        """
        if not reviews:
            return {}

        ratings = []
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}

        for review in reviews:
            try:
                result = self.analyze_sentiment(review)
                ratings.append(result['rating'])

                if result['sentiment'] == 'positive':
                    sentiment_counts['positive'] += 1
                elif result['sentiment'] == 'neutral':
                    sentiment_counts['neutral'] += 1
                else:
                    sentiment_counts['negative'] += 1
            except Exception as e:
                logger.warning(f"Error processing review: {str(e)}")
                continue

        total = len(ratings)
        if total == 0:
            return {}

        return {
            'avg_rating': sum(ratings) / total,
            'positive_pct': (sentiment_counts['positive'] / total) * 100,
            'neutral_pct': (sentiment_counts['neutral'] / total) * 100,
            'negative_pct': (sentiment_counts['negative'] / total) * 100,
            'total_reviews': total
        }

    def evaluate_model(self, X_test, y_test) -> Dict[str, float]:
        """
        Evaluate model performance on test data

        Args:
            X_test: Test features
            y_test: True labels

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if not self.model:
            raise ValueError("Model not loaded. Please load the model first.")

        y_pred = self.model.predict(X_test)

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted')
        }


# Example usage
if __name__ == '__main__':
    # Initialize analyzer with default paths
    analyzer = SentimentAnalyzer(
        model_path="../models/sentiment_model_fr.pkl",
        vectorizer_path="../models/tfidf_vectorizer_fr.pkl"
    )

    # Test prediction
    test_reviews = [
        "Service excellent, je suis très satisfait!",
        "Produit correct mais pourrait être amélioré",
        "Très déçu par la qualité, je ne recommande pas",
        "Livraison rapide et produit conforme à mes attentes",
        "Médiocre, ne correspond pas à la description",
        "Excellent. Très bonne banque !Remarque les esprits chagrins qui mettent 1 étoile à wafabank  doivent savoir que wafabank meilleure banque au Maroc et plus n'accueille pas n'importe qui avec le sourire pourdes personnes peu claires avec des rentrées d'argent hypothétiques et des projets fumeux."
    ]

    for review in test_reviews:
        result = analyzer.analyze_sentiment(review)
        print(f"Review: {review[:50]}...")
        print(f"Rating: {result['rating']} stars")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("-" * 50)

    # Get batch statistics
    stats = analyzer.get_sentiment_stats(test_reviews)
    print("\nBatch Statistics:")
    print(f"Average Rating: {stats['avg_rating']:.2f} stars")
    print(f"Positive: {stats['positive_pct']:.1f}%")
    print(f"Neutral: {stats['neutral_pct']:.1f}%")
    print(f"Negative: {stats['negative_pct']:.1f}%")
