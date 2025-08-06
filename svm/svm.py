import numpy as np
from sklearn.svm import SVC  # ⚙️ Support Vector Classifier from scikit-learn

class SVMModel:
    def __init__(self, kernel="linear", C=1.0):
        """
        🔧 Initializes the SVM model.
        
        Parameters:
        - kernel: Kernel type ('linear', 'rbf', 'poly', etc.)
        - C: Regularization parameter (default 1.0)
        """
        self.model = SVC(kernel=kernel, C=C)

    def train(self, X, y):
        """
        🏋️‍♂️ Train the SVM model on features X and labels y
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        🔮 Predict class labels for the given input X
        """
        return self.model.predict(X)
