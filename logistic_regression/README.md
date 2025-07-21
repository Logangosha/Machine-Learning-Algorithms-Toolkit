# 🎯 Logistic Regression

This module implements a wrapper for **Logistic Regression** using `scikit-learn`. It allows for binary classification tasks like predicting if a tumor is malignant or benign.

---

## 🧠 What is Logistic Regression?

Logistic Regression is a **supervised machine learning algorithm** used for classification. It outputs a probability between 0 and 1, which is then thresholded to classify input data into two categories.

### Equation:
P(y=1) = 1 / (1 + e^-(w·x + b))

Where:
- `w` = weights  
- `x` = input features  
- `b` = intercept  
- `P(y=1)` = predicted probability of class 1

---

## 📦 What's Inside

- init.py
- logistic_regression.py
- example.ipynb
- README.md

---

## 🔍 Quick Example

```python
from logistic_regression import LogisticRegressionModel
import numpy as np

# Sample dataset: 2 features per point
X = np.array([[2, 3], [1, 5], [2, 1], [4, 2]])
y = np.array([1, 1, 0, 0])

model = LogisticRegressionModel()
model.train(X, y)

prediction = model.predict([[3, 2]])
print(prediction)  # → [0] or [1]
```

## 📊 Evaluation Metrics
You can use:
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Visualized in the included notebook!

 ## 📚 Dependencies
- numpy
- pandas
- matplotlib
- scikit-learn