import sys
sys.path.append('Lab_04')

from sklearn.model_selection import train_test_split
from src.models.text_classifier import TextClassifier
from src.models.vectorizer import Vectorizer

def main():
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad."
    ]
    labels = [1, 0, 1, 0, 1, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, train_size=0.8, random_state=42
    )

    vectorizer = Vectorizer(max_features=1000)

    classifier = TextClassifier(vectorizer=vectorizer)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    metrics = classifier.evaluate(y_test, y_pred)

    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()