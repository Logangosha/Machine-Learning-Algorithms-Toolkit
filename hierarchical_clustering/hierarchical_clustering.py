import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

class HierarchicalClustering:
    def __init__(self, method="ward", metric="euclidean"):
        """
        Initialize the hierarchical clustering model.

        Parameters:
        ----------------
        method : str, default="ward"
            🔸 This defines **how distances between clusters are computed**.
            🔸 Options include:
                - 'ward'     : Minimizes the variance within clusters (most common).
                - 'single'   : Minimum distance between points (can cause "chaining").
                - 'complete' : Maximum distance between points.
                - 'average'  : Average distance between all points in two clusters.

        metric : str, default="euclidean"
            🔸 This defines the **distance metric between individual points**.
            🔸 Common options:
                - 'euclidean' : Straight-line distance (default, great for many tasks).
                - 'manhattan' : Block distance (L1 norm).
                - 'cosine'    : Cosine angle similarity.
                - ... more available in SciPy docs.

        Example:
        --------
        model = HierarchicalClustering(method="average", metric="cosine")
        """
        self.method = method
        self.metric = metric
        self.linkage_matrix = None  # This will store the result of linkage()

    def fit(self, X):
        """
        Compute the linkage matrix for the given data.

        Parameters:
        ----------------
        X : ndarray of shape (n_samples, n_features)
            🔸 The input data you want to cluster.
            🔸 Each row = one sample. Each column = a feature.

        Output:
        ----------------
        self.linkage_matrix : ndarray
            🔸 Stores the hierarchy of merges.
            🔸 Required for further steps like plotting or cluster assignment.
        """
        self.linkage_matrix = linkage(X, method=self.method, metric=self.metric)

    def get_clusters(self, n_clusters):
        """
        Assign cluster labels to each sample by cutting the dendrogram.

        Parameters:
        ----------------
        n_clusters : int
            🔸 Number of clusters you want to extract from the hierarchy.
            🔸 This determines where the dendrogram is cut.

        Returns:
        ----------------
        labels : ndarray of shape (n_samples,)
            🔸 Each entry is a cluster ID (1-based).
            🔸 For example: [1, 2, 1, 3, 2, 1]

        Example:
        --------
        labels = model.get_clusters(n_clusters=3)
        """
        if self.linkage_matrix is None:
            raise ValueError("Must call `fit()` before getting clusters.")

        return fcluster(self.linkage_matrix, n_clusters, criterion="maxclust")
