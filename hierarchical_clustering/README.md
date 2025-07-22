# 🌳 Hierarchical Clustering

This module implements **Hierarchical Clustering** using `scikit-learn`. It helps group similar data points into clusters by building a tree-like structure (called a dendrogram) based on distances between data points.

---

## 🧠 What is Hierarchical Clustering?

Hierarchical Clustering is an **unsupervised learning algorithm** that either:

- 🔼 **Agglomerative** (bottom-up): Each point starts in its own cluster, then they merge step by step.
- 🔽 **Divisive** (top-down): Start with one big cluster, then split it down.

👉 We use the **Agglomerative** approach in this module.

📺 [Watch this 6-min video explanation](https://www.youtube.com/watch?v=8QCBl-xdeZI)

---

## ⚙️ Parameters

The main model is based on `sklearn.cluster.AgglomerativeClustering`.

You can configure it like this:

```python
HierarchicalClusteringModel(
    n_clusters=2,         # Number of clusters to form (default = 2)
    affinity='euclidean', # Metric for distance (e.g., 'euclidean', 'manhattan')
    linkage='ward'        # How to merge clusters: 'ward', 'complete', 'average', or 'single'
)
```
---

## 📦 What's Inside
- init.py
- hierarchical_clustering.py
- example.ipynb
- README.md

---

## 🧪 Quick Example
```python
from hierarchical_clustering import HierarchicalClusteringModel
import numpy as np

# Sample 2D data
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8]])

model = HierarchicalClusteringModel(n_clusters=2)
labels = model.fit_predict(X)

print(labels)  # e.g., [1 1 0 0]
```

---

## 🌿 Visual Output
Dendrograms help show how clusters were formed step-by-step.
The included example.ipynb walks through how to plot one using:
```python
from scipy.cluster.hierarchy import dendrogram, linkage
```
---

## 📦 What's Inside
- numpy
- matplotlib
- scikit-learn
- scipy 

---

---

## 🔗 Use Cases
- 📊 Customer segmentation
- 🧬 Gene expression analysis
- 📄 Document clustering
- 🤝 Social network analysis

---