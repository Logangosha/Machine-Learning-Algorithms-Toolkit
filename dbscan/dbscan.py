# dbscan.py

from sklearn.cluster import DBSCAN

class DBSCANCluster:
    def __init__(self, eps=0.5, min_samples=5):
        """
        📦 eps: Radius around a point to define a neighborhood
        🧩 min_samples: Minimum points needed to form a dense region
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def fit_predict(self, X):
        """
        Fits DBSCAN to the data and returns cluster labels.
        
        🧠 Returns:
        - An array of cluster labels (noise points are labeled -1)
        """
        return self.model.fit_predict(X)

    def core_sample_indices(self):
        """
        🔍 Returns indices of core samples.
        """
        return self.model.core_sample_indices_

    def components(self):
        """
        💠 Returns the core samples used in clustering.
        """
        return self.model.components_
