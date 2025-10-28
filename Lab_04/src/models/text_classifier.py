from src.models.vectorizer import Vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TextClassifier:
    def __init__(self, vectorizer: Vectorizer):
        self.vectorizer = vectorizer
        self._model = None

    def fit(self, texts: list[str], labels: list[int]):
        X = self.vectorizer.fit_transform(texts)

        self._model = LogisticRegression(solver='liblinear', random_state=42)

        self._model.fit(X, labels)

    def predict(self, texts: list[str]) -> list[str]:
        if (self._model is None):
            return None
        
        X = self.vectorizer.transform(texts)

        return self._model.predict(X).tolist()
    
    def evaluate(self, y_true: list[int], y_pred: list[int]) -> dict[str, float]:
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0)
        }