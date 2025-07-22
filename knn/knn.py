import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier:
    def __init__(self, n_neighbors=3):
        """
        Initialize the KNN model with a given number of neighbors.
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X, y):
        """
        Fit the KNN model to the training data.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the class labels for the input data.
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        """
        return self.model.score(X, y)
