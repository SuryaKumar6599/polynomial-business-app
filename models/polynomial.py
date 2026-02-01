import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score


class PolynomialRegressionModel:
    def __init__(self, degree=2):
        self.degree = degree
        self.model = None

    def _build_pipeline(self):
        return Pipeline([
            ("poly", PolynomialFeatures(self.degree, include_bias=False)),
            ("lr", LinearRegression())
        ])

    def fit(self, X, y):
        self.model = self._build_pipeline()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return {
            "r2": self.model.score(X, y),
            "rmse": np.sqrt(np.mean((y - preds) ** 2)),
            "mae": np.mean(np.abs(y - preds))
        }

    @staticmethod
    def auto_select_degree(X, y, max_degree=5, cv=5):
        scores = {}
        for d in range(1, max_degree + 1):
            pipe = Pipeline([
                ("poly", PolynomialFeatures(d, include_bias=False)),
                ("lr", LinearRegression())
            ])
            scores[d] = cross_val_score(pipe, X, y, cv=cv).mean()

        best_degree = max(scores, key=scores.get)
        return {"best_degree": best_degree, "cv_scores": scores}


class RidgePolynomialRegressionModel(PolynomialRegressionModel):
    def __init__(self, degree=2, alpha=1.0):
        self.degree = degree
        self.alpha = alpha
        self.model = None

    def _build_pipeline(self):
        return Pipeline([
            ("poly", PolynomialFeatures(self.degree, include_bias=False)),
            ("ridge", Ridge(alpha=self.alpha))
        ])


class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
