# 🌲 XGBoost Regression & Classification

This project demonstrates how to use **XGBoost** (Extreme Gradient Boosting) for both **regression** (e.g., predicting house prices) and **classification** (e.g., predicting if a house is expensive) tasks using synthetic data.

---

## 🤔 What is XGBoost?

[XGBoost explained (video)](https://www.youtube.com/watch?v=XWQ0Fd_xiBE)

XGBoost is a **powerful, scalable machine learning algorithm** based on gradient-boosted decision trees.

It’s designed for:
- Accuracy 🥇  
- Speed ⚡  
- Flexibility 🔧  

---

## 📁 What's Included

- `xgboost_module.py` — Lightweight wrapper around `xgboost.XGBRegressor` and `xgboost.XGBClassifier` with training & prediction methods  
- `example_regression.ipynb` — Full walkthrough: synthetic data generation, training, evaluation, and plotting for regression  
- `example_classification.ipynb` — Similar walkthrough for classification with metrics like accuracy and confusion matrix  
- `README.md` — You’re reading it!

---

## 🧪 Quick Usage (Regression)

```python
from xgboost import XGBoostModel
import numpy as np

# Simple input features and target prices
X = np.array([[1000], [1500], [2000]])
y = np.array([[200000], [300000], [400000]])

model = XGBoostModel(task="regression")
model.train(X, y)

# Predict price for a 1800 sq ft home
print(model.predict([[1800]]))  # ~360000 (approx)
```
---

## 📊 Evaluation Metrics Explained

- **Mean Squared Error (MSE):**  
  Measures the average of the squared differences between predicted and actual values. Squaring penalizes larger errors more heavily and avoids positive/negative error cancellation. Lower MSE means better predictions.

- **R² Score:**  
  Represents how well the model explains the variance in the data.  
  1.0 = perfect prediction, 0 = model predicts no better than the mean.

- **Classification Metrics (in classification example):**  
  Accuracy, confusion matrix, precision, recall, and F1 score.

---

## 📚 Dependencies

- xgboost  
- numpy  
- pandas  
- matplotlib  
- scikit-learn  