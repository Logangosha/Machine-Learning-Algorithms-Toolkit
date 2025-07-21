import numpy as np
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self):
        # Initialize a basic logistic regression model from sklearn
        self.model = LogisticRegression()

    def train(self, X, y):
        """
        Train the model using features X and labels y.
        X: 2D array (samples x features)
        y: 1D array (binary labels: 0 or 1)
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict class labels (0 or 1) for input features X.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities instead of class labels.
        Returns: probability of class 0 and 1 for each input.
        """
        return self.model.predict_proba(X)

    def coefficients(self):
        """
        Return the model's learned weights and intercept.
        """
        return self.model.coef_, self.model.intercept_
