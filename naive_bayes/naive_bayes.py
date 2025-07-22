import numpy as np
from sklearn.naive_bayes import GaussianNB

class NaiveBayesModel:
    """
    Naive Bayes classifier using scikit-learn's GaussianNB.
    Suitable for classification problems with continuous features.
    """
    def __init__(self):
        # Initialize the Gaussian Naive Bayes model
        self.model = GaussianNB()

    def train(self, X, y):
        """
        Fit the model to the training data.
        X: feature matrix (numpy array)
        y: labels (numpy array)
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the labels for new data.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for new data.
        """
        return self.model.predict_proba(X)

    def accuracy(self, X, y):
        """
        Return the classification accuracy on a test set.
        """
        return self.model.score(X, y)
