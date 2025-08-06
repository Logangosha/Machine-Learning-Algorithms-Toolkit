

# 🧪 DBSCAN Clustering with sklearn

This mini project demonstrates **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** using `sklearn`, with a simple wrapper and example notebook.

---

## 📚 What is DBSCAN?

- https://www.youtube.com/watch?v=_A9Tq6mGtLI
- https://www.youtube.com/watch?v=Lh2pAkNNX1g

**DBSCAN** is a clustering algorithm that groups together points that are **close to each other** (within `eps` distance) and marks points in low-density regions as **noise**.

✅ Good for non-spherical clusters  
❌ Not great for clusters of varying density  

---

## 📦 What’s Inside

- init.py
- dbscan.py
- example.ipynb
- example_2.ipynb
- README.md

---

## 🚀 How It Works

```python
from dbscan import DBSCANCluster
import numpy as np

# Example: Moon-shaped data
from sklearn.datasets import make_moons
X, _ = make_moons(n_samples=300, noise=0.1)

model = DBSCANCluster(eps=0.3, min_samples=5)
labels = model.fit_predict(X)
```

- eps: Distance to define neighborhood

- min_samples: Minimum neighbors to form a cluster

Use .core_sample_indices_ or .components_() to dive deeper.

---

## 📊 Visualization
Clusters and noise are shown using matplotlib.
Try adjusting eps or min_samples and watch the changes!

---

## 📚 Dependencies
- numpy
- matplotlib
- scikit-learn

🧠 Fun fact: DBSCAN doesn’t need you to choose the number of clusters upfront — it figures it out from the data! 🙌