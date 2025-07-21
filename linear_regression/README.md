# 📈 Linear Regression

This module implements **Linear Regression** from scratch using only `numpy`. It can be used to predict a continuous value (like house prices) based on one or more input features (like square footage, bedrooms, etc.).

---

## 🧠 What is Linear Regression?

Linear Regression is a **supervised machine learning algorithm** that models the relationship between input variables (**X**) and a continuous output (**y**) by fitting a line (or plane) that minimizes the prediction error.

### Equation (Multiple Features):

y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

Where:
- `x₁, x₂, ..., xₙ` = input features  
- `w₁, w₂, ..., wₙ` = learned weights (slopes)  
- `b` = bias/intercept  
- `y` = predicted output

---

## 📦 What's Inside

- init.py
- linear_regression.py
- example.ipynb
- README.md

---

## 🧪 Quick Example

```python
from linear_regression import LinearRegressionModel
import numpy as np

# Simple dataset: square footage vs price
X = np.array([[1000], [1500], [2000]])
y = np.array([[200000], [300000], [400000]])

model = LinearRegressionModel()
model.train(X, y)

# Predict price for a 1800 sq ft home
prediction = model.predict([[1800]])
print(prediction)  # → [[360000]] (approx)
```

## 📊 Evaluation
You can evaluate model accuracy using:

- Mean Squared Error (MSE): Measures average squared difference between actual and predicted values.
- R² Score: Tells how well the data fits the model (1.0 is perfect).

These metrics are demonstrated in the example.ipynb notebook with plots.

## 📚 Dependencies
Only uses:

- numpy
- pandas
- matplotlib
- scikit-learn