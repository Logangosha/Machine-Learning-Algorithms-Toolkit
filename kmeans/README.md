# 🔍 K-Means Clustering

This module provides a **KMeansModel** class that wraps around scikit‑learn’s efficient `KMeans` algorithm. It enables easy clustering of data into **k groups** based on feature similarity.

---

## 🧠 What is K‑Means?

**K‑Means** is a classic **unsupervised machine learning** method that:

1. Initializes `k` cluster centers.
2. Assigns each data point to the nearest center.
3. Updates centers as the centroid of assigned points.
4. Repeats steps 2–3 until convergence.

Common use cases:
- Customer segmentation
- Image & document clustering
- Anomaly detection

---

## 🧪 Quick Example

```python
from kmeans import KMeansModel
import numpy as np

X = np.array([
    [1, 2], [1, 4], [1, 0],
    [4, 2], [4, 4], [4, 0]
])

model = KMeansModel(n_clusters=2)
model.fit(X)
labels = model.predict(X)

print(labels)  # e.g. [1 1 1 0 0 0]
```
---

## 📦 Package Contents
- __init__.py
- kmeans.py
- example.ipynb
- README.md

---

## 🔧 Parameters
```python
KMeansModel(
    n_clusters=3,
    max_iters=300,
    random_state=42
)
```
- n_clusters: # of clusters
- max_iters: max iterations
- random_state: reproducible results

---

## 📚 Dependencies
- numpy
- matplotlib (for example notebook)
- scikit-learn

---

## 🔗 References
- scikit‑learn’s KMeans docs:
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

- Demo walkthrough video:
YouTube: K‑Means clustering explained

---

