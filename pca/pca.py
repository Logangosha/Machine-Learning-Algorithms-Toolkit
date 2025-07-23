import numpy as np
from sklearn.decomposition import PCA

class PrincipalComponentAnalysis:
    """
    A wrapper class around sklearn's PCA to simplify usage and keep things modular.
    """

    def __init__(self, n_components=None):
        """
        Initialize the PCA model.
        
        Parameters:
        - n_components (int or None): Number of principal components to keep.
                                      If None, all components are kept.
        """
        self.n_components = n_components
        self.model = PCA(n_components=n_components)

    def fit(self, X):
        """
        Fit the PCA model on the dataset X.
        
        Parameters:
        - X (ndarray): The input data matrix (rows = samples, columns = features).
        """
        # Center the data and learn the principal components
        self.model.fit(X)

    def transform(self, X):
        """
        Project the data onto the principal components.
        
        Parameters:
        - X (ndarray): The input data matrix.
        
        Returns:
        - Transformed data in lower-dimensional space.
        """
        return self.model.transform(X)

    def fit_transform(self, X):
        """
        Fit the PCA model and return the transformed data.
        
        Parameters:
        - X (ndarray): The input data matrix.
        
        Returns:
        - Reduced dimensional data.
        """
        return self.model.fit_transform(X)

    def explained_variance_ratio(self):
        """
        Get the percentage of variance explained by each principal component.
        
        Returns:
        - ndarray: Variance ratio of each component.
        """
        return self.model.explained_variance_ratio_

    def components(self):
        """
        Get the principal axes in feature space.
        
        Returns:
        - ndarray: Principal components (eigenvectors).
        """
        return self.model.components_

    def singular_values(self):
        """
        Get the singular values corresponding to each principal component.
        
        Returns:
        - ndarray: Singular values.
        """
        return self.model.singular_values_

    def mean(self):
        """
        Get the per-feature empirical mean.
        
        Returns:
        - ndarray: Mean of each feature.
        """
        return self.model.mean_
