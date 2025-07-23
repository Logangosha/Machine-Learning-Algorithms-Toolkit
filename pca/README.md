# 📊 Principal Component Analysis (PCA)

This module provides a simple, modular wrapper around `scikit-learn`'s PCA implementation to perform **dimensionality reduction** on datasets.

---

## 🧠 What is PCA?

Principal Component Analysis (PCA) is an **unsupervised learning technique** that transforms high-dimensional data into a lower-dimensional space by finding the directions (principal components) that maximize the variance in the data.

### Why use PCA?
- Reduce the number of features while retaining most information.
- Visualize high-dimensional data in 2D or 3D.
- Speed up other machine learning algorithms by simplifying input.

---

## 📐 How PCA Works (Simplified)

1. Center the data by subtracting the mean of each feature.
2. Compute the covariance matrix.
3. Find eigenvectors (principal components) and eigenvalues of the covariance matrix.
4. Project data onto the top principal components that explain the most variance.

---

## 📦 What’s Inside

- init.py
- pca.py
- example.ipynb
- README.md

---

## 🧪 Quick Usage Example

```python
import numpy as np
from pca import PrincipalComponentAnalysis

# Sample data: 5 samples, 3 features
X = np.array([
    [2.5, 2.4, 0.5],
    [0.5, 0.7, 2.1],
    [2.2, 2.9, 1.9],
    [1.9, 2.2, 1.8],
    [3.1, 3.0, 2.7]
])

# Initialize PCA to reduce to 2 components
pca = PrincipalComponentAnalysis(n_components=2)

# Fit and transform data
X_reduced = pca.fit_transform(X)

print("Reduced data shape:", X_reduced.shape)
print("Explained variance ratio:", pca.explained_variance_ratio())
```

---

## 📚 Dependencies

- numpy
- scikit-learn
- matplotlib (for example.ipynb only)
- pandas (optional, for example.ipynb)

---

## 🤓 Fun Fact
PCA was invented over 100 years ago (1901) by Karl Pearson! It’s still one of the most popular techniques for simplifying complex datasets.

---

## 🚀 Get Started!
Try running the example.ipynb notebook to see PCA in action on the famous Iris dataset!

