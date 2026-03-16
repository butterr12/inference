from typing import Optional

import config


class SentimentAnalyzer:
    def __init__(self, threshold: float = None):
        self.sentiment_labels = config.SENTIMENT_LABELS
        self.vector_labels = config.VECTOR_LABELS
        self.threshold = threshold or config.VECTOR_THRESHOLD

    def parse_response(self, data: dict) -> tuple:
        if data is None:
            return [], []

        labels = [p.get("label", "") for p in data]
        scores = [p.get("score", 0.0) for p in data]
        return labels, scores

    def get_best_sentiment(self, labels: list, scores: list) -> tuple:
        best_score = 0.0
        best_label = "neutral"

        for label, score in zip(labels, scores):
            if label in self.sentiment_labels and score > best_score:
                best_label = label
                best_score = score

        return best_label, best_score

    def get_vectors(self, labels: list, scores: list) -> str:
        selected = [
            label for label, score in zip(labels, scores)
            if label in self.vector_labels and score >= self.threshold
        ]
        return ", ".join(selected)

    def analyze(self, api_result: dict) -> Optional[dict]:
        if api_result.get("error"):
            return None

        data = api_result.get("data")
        if isinstance(data, dict) and "error" in data:
            return None

        labels, scores = self.parse_response(data)
        if not labels:
            return None

        sentiment_label, sentiment_score = self.get_best_sentiment(labels, scores)
        vectors = self.get_vectors(labels, scores)

        return {
            "sentimentLabel": sentiment_label,
            "score": float(sentiment_score),
            "vectors": vectors
        }
