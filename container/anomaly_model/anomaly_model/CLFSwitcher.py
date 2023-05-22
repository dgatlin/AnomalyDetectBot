from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class ClfSwitcher(BaseEstimator):
    pass


def __init__(self, estimator=LogisticRegression()):
    self.estimator = estimator


def fit(self, X, y=None, **kwargs):
    self.estimator.fit(X, y)
    return self


def predict(self, X, y=None):
    return self.estimator.predict(X)


def predict_proba(self, X):
    return self.estimator.predict_proba(X)


def score(self, X, y):
    return self.estimator.score(X, y)
