import numpy as np
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeModel:
    def __init__(self, max_depth=None, random_state=42):
        """
        Initialize the Decision Tree Classifier.

        Parameters:
        - max_depth: Maximum depth of the tree.
        - random_state: Controls randomness for reproducibility.
        """
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

    def train(self, X, y):
        """
        Train the decision tree model.

        Parameters:
        - X: Input features (2D array)
        - y: Target labels (1D or 2D array)
        """
        self.model.fit(X, y.ravel())  # ravel in case y is column vector

    def predict(self, X):
        """
        Predict using the trained decision tree.

        Parameters:
        - X: Input features

        Returns:
        - Predictions
        """
        return self.model.predict(X)

    def get_feature_importance(self):
        """
        Returns importance of each feature used by the decision tree.
        """
        return self.model.feature_importances_
