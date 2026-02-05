import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib

class Trainer():
    def __init__(self, preprocessor):
        self.model_pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000))
        ])

    def train(self, x_train, y_train, x_test, y_test):
        self.model_pipe.fit(x_train, y_train)

        y_pred = self.model_pipe.predict(x_test)

        stats = classification_report(y_test, y_pred)

        joblib.dump(self.model_pipe, "artifacts/model.pkl")

        train_sizes, train_scores, val_scores = learning_curve(
            self.model_pipe, x_train, y_train, cv=5, scoring="accuracy"
        )


        plt.plot(train_sizes, train_scores.mean(axis=1), label="train")
        plt.plot(train_sizes, val_scores.mean(axis=1), label="validation")
        plt.legend()
        plt.savefig("artifacts/learning_curve.png", dpi=300, bbox_inches="tight")
        plt.show()

        return stats
