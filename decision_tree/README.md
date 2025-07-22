# 🌳 Decision Tree Classifier

This module wraps the `sklearn` **DecisionTreeClassifier** for easy reuse and learning. Decision Trees are intuitive, powerful classifiers that split data based on feature thresholds to make predictions.

---

## 🧠 What is a Decision Tree?

A **Decision Tree** is a supervised learning algorithm used for classification and regression. It works by asking yes/no questions to split the data into subsets.

### Example Rule:
Is Age > 30?
Yes: Predict "Not Spam"
No:  Predict "Spam"

**They’re great when:**
- You want interpretability ✅
- Your data has non-linear boundaries ✅
- You need a quick model without heavy preprocessing ✅

---

## 📦 What's Inside
- __init__.py
- decision_tree.py
- example.ipynb 
- README.md 

---

## 🚀 Quick Usage\
```python
from decision_tree import DecisionTreeModel
import numpy as np

# Sample input (2 features)
X = np.array([[5, 1], [10, 0], [2, 1]])
y = np.array([1, 0, 1])  # Binary class labels

model = DecisionTreeModel(max_depth=3)
model.train(X, y)

# Make predictions
preds = model.predict([[6, 1]])
print(preds)  # → [1]
```
---

## 📚 Dependencies
- numpy
- scikit-learn