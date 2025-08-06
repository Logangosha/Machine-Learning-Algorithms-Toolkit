# 🧠 Support Vector Machine (SVM) Classifier

A minimal, clean SVM classifier demo using `scikit-learn`! Great for binary classification problems. Uses synthetic data for demonstration.

---

## 📚 What is SVM?

https://www.youtube.com/watch?v=N-sPSXDtcQw

**Support Vector Machine** is a supervised learning algorithm used for classification and regression. It finds the best **hyperplane** that separates data points of different classes.

### ✏️ Key Idea:
SVM tries to maximize the **margin** between classes. The data points closest to the margin are called **support vectors**.

---

## 📦 What’s Inside

- init.py
- svm.py
- example.ipynb
- README.md

---

## 🚀 How to Use

```python
from svm import SVMModel

model = SVMModel(kernel="linear")
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

### You can evaluate using:

- ✅ Accuracy Score
- 📊 Classification Report

---

## 📉 Visual Output

The notebook includes a decision boundary plot to help see how SVM separates the classes.

---

## 📚 Dependencies
- numpy
- matplotlib
- scikit-learn