# 🌲 Random Forest Regressor

This project shows how to use **Random Forest Regression** from `scikit-learn` to predict house prices using synthetic data.

---

## 🧠 What is Random Forest?

https://www.youtube.com/watch?v=cIbj0WuK41w

Random Forest is an **ensemble learning algorithm** that builds multiple decision trees and averages their predictions. It is:
- Accurate
- Resistant to overfitting
- Great for both classification and regression tasks

---

## 📦 What’s Inside

- init.py
- random_forest.py
- example.ipynb
- README.md

---

## 🚀 Quick Example

```python
from random_forest import RandomForestModel
import numpy as np

X = np.array([[1000], [1500], [2000]])
y = np.array([200000, 300000, 400000])

model = RandomForestModel()
model.train(X, y)

prediction = model.predict([[1800]])
print(prediction)  # e.g., [360500.0]
```

---

## 📊 Evaluation
This project uses:
- Mean Squared Error (MSE) to measure average error
- R² Score to measure model accuracy (closer to 1 is better)

---

## 📚 Dependencies
- numpy
- matplotlib
- scikit-learn
- pandas (optional)

