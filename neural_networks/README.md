# 🤖 Neural Networks with scikit-learn: Regression & Classification

This project demonstrates using scikit-learn's neural networks (MLP) for **both** regression and classification tasks.

---

## 🧠 What is an MLP?

- Multi-Layer Perceptron (MLP) is a neural network model that can perform:
  - **Classification:** Assign inputs to discrete categories
  - **Regression:** Predict continuous values

---

## 🚀 Usage

The `NeuralNetworkModel` class supports two tasks, controlled by the `task` parameter:

- `task='classification'` for classification problems  
- `task='regression'` for regression problems

---

## 📦 Example: Regression

```python
from neural_networks import NeuralNetworkModel

# Create and train regression model
reg_model = NeuralNetworkModel(task='regression', hidden_layer_sizes=(50,), max_iter=300)
reg_model.train(X_train, y_train)

# Predict continuous outputs
y_pred = reg_model.predict(X_test)
```

---

## 📦 Example: Classification

```python
from neural_networks import NeuralNetworkModel

# Create and train classification model
clf_model = NeuralNetworkModel(task='classification', hidden_layer_sizes=(50,), max_iter=300)
clf_model.train(X_train, y_train)

# Predict class labels
y_pred = clf_model.predict(X_test)
```

---

## 📊 Evaluation

- **Regression:**
  - Mean Squared Error (MSE)
  - R² Score
- **Classification:**
  - Accuracy
  - Precision, Recall, F1-score (via classification report)

---

## 📚 Dependencies

- numpy  
- matplotlib  
- scikit-learn  

---

## 🎯 Productivity Tip

Switch tasks easily by toggling the `task` parameter — no need to change your pipeline drastically!

---

## 🤓 Fun Fact

MLPs are "universal approximators" — with enough neurons and layers, they can model virtually any function! 🔥
