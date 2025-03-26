import re
from typing import Optional
import spacy
from nltk.corpus import stopwords
from emot.emo_unicode import EMOTICONS_EMO


class TextPreprocessor:
    def __init__(self, language: str = "fr"):
        self.nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
        self.stopwords = set(stopwords.words(language))
        self.custom_stopwords = {"banque", "agence", "client", "..."}

    def _handle_emoticons(self, text: str) -> str:
        for emot, meaning in EMOTICONS_EMO.items():
            text = re.sub(re.escape(emot), f" {meaning} ", text)
        return text

    def _lemmatize(self, text: str) -> str:
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def clean(self, text: str) -> str:
        text = text.lower()
        text = self._handle_emoticons(text)
        text = re.sub(r'http\S+|www\S+|@\w+', '', text)  # Remove URLs/mentions
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = self._lemmatize(text)
        tokens = [word for word in text.split() if word not in self.stopwords]
        return " ".join(tokens)