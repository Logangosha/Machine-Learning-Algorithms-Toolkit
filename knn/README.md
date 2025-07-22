# 🎯 K-Nearest Neighbors (KNN)

This module demonstrates **K-Nearest Neighbors classification** using simple, synthetic 2D data. Great for learning and visualization!

---

## 🧠 What is KNN?

KNN is a **supervised classification algorithm** that:
1. Stores the training data.
2. Classifies new data by finding the **k nearest neighbors** and voting on the label.

💡 It’s simple and powerful — no training phase required.

---

## 📦 What’s Inside

- init.py
- knn.py
- example.ipynb
- README.md

---

## 🚀 Quick Example

```python
from knn import KNNClassifier
import numpy as np

# Make some fake 2D points
X = np.array([[1, 1], [2, 2], [6, 6], [7, 7]])
y = np.array([0, 0, 1, 1])  # Two classes

model = KNNClassifier(n_neighbors=3)
model.train(X, y)

# Predict new point's class
print(model.predict([[3, 3]]))  # → likely class 0
```
📊 What's Visualized?
In example.ipynb, we:

- Generate 2D clustered data for 2 classes
- Train a KNN classifier
- Plot new points and their predicted class
- Show how changing the position changes the prediction

📚 Dependencies
- numpy
- matplotlib
- scikit-learn

🔧 Tips for Playing
Try changing:

- The number of samples
- k in the classifier
- The cluster centers
- Add a third class 🎨