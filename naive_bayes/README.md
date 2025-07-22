# 🌸 Naive Bayes

This module implements a **Naive Bayes Classifier** using `scikit-learn`'s `GaussianNB`. It's great for classification tasks, especially when features are continuous and normally distributed.

---

## 🧠 What is Naive Bayes?

Naive Bayes is a **probabilistic classification algorithm** based on Bayes’ Theorem with an assumption of independence between features.

### Bayes’ Theorem:

P(A|B) = (P(B|A) * P(A)) / P(B)

It calculates the probability of each class and picks the most likely one for a given input.

---

## 📦 What's Inside
  
__init__.py  
naive_bayes.py
example.ipynb
README.md

---

## 🧪 Quick Example

```python
from naive_bayes import NaiveBayesModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = NaiveBayesModel()
model.train(X_train, y_train)

# Predict
predictions = model.predict(X_test)
print(predictions)
```

## 📊 Evaluation
Use accuracy, confusion matrix, and classification report from sklearn.metrics for evaluation.
The included example notebook shows all this with plots and real output ✅

## 📚 Dependencies
- numpy
- pandas
- matplotlib
- scikit-learn