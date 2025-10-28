from sklearn.feature_extraction.text import TfidfVectorizer

class Vectorizer:
    def __init__(self, max_features: int = 5000, ngram_range=(1, 1)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )

    def fit_transform(self, texts: list[str]):
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts: list[str]):
        return self.vectorizer.transform(texts)