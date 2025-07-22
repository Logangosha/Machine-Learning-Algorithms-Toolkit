import numpy as np
from sklearn.cluster import KMeans

class KMeansModel:
    """
    Wrapper around sklearn's KMeans clustering algorithm.

    Parameters:
    -----------
    n_clusters : int
        Number of clusters to form.
    max_iters : int
        Maximum number of iterations for convergence.
    random_state : int
        Seed for reproducibility.

    Attributes:
    -----------
    centers : np.ndarray
        Coordinates of cluster centers after fitting.
    """

    def __init__(self, n_clusters=3, max_iters=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters,
                            max_iter=max_iters,
                            random_state=random_state)
        self.centers = None

    def fit(self, X):
        """
        Fit the KMeans model on data X.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data points to cluster.

        Process:
        --------
        Uses sklearn's KMeans fit method to find cluster centers and assign labels.
        """
        self.model.fit(X)
        self.centers = self.model.cluster_centers_

    def predict(self, X):
        """
        Predict cluster labels for new data points using the trained model.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns:
        --------
        labels : np.ndarray, shape (n_samples,)
            Predicted cluster indices.
        """
        if self.centers is None:
            raise Exception("Model has not been fitted yet. Call 'fit' first.")
        return self.model.predict(X)
