# neural_networks.py
# 🌟 Educational Neural Network Wrapper 🌟
# Purpose: Provide a reusable, easy-to-understand class for training
# and using neural networks for both classification and regression.

# 🔍 What is a Neural Network?
# A neural network is a machine learning model inspired by the human brain 🧠.
# It's made up of *layers* of *neurons* (aka nodes), which are simple mathematical units
# that process data and learn patterns.

# Each connection has a "weight" and each neuron applies an "activation function"
# to introduce non-linearity and make learning more powerful.

# Neural networks learn by *backpropagation* – adjusting the weights
# to minimize error between predictions and actual outputs over many iterations (epochs).

# Neural nets can be used for:
# 🧮 Regression – predicting numbers (e.g. price, temperature)
# 🧾 Classification – predicting categories (e.g. spam vs not spam)

# ------------------------------
# 📦 Import models from sklearn
# ------------------------------
from sklearn.neural_network import MLPRegressor, MLPClassifier
# MLP = Multi-Layer Perceptron
# These are high-level neural network implementations provided by scikit-learn.

# ------------------------------
# 🧠 NeuralNetworkModel Class
# ------------------------------
class NeuralNetworkModel:
    def __init__(self, task='classification', hidden_layer_sizes=(100,), max_iter=200):
        """
        Initialize the neural network model.

        Parameters:
        - task: str — Choose 'classification' or 'regression'.
            Determines whether we're using MLPClassifier or MLPRegressor.
        - hidden_layer_sizes: tuple — Architecture of the hidden layers.
            Example: (100,) means 1 hidden layer with 100 neurons.
                     (64, 32) means 2 hidden layers with 64 and 32 neurons.
        - max_iter: int — Number of iterations (epochs) to train the model.
            More iterations can improve accuracy but take more time.
        """

        self.task = task  # Save the type of task (for reference/debugging)

        # ------------------------------
        # 🏗️ Build the model based on task
        # ------------------------------
        if task == 'regression':
            # MLPRegressor: for continuous output predictions (e.g., house prices)
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,  # neural net structure
                max_iter=max_iter,                      # training epochs
                random_state=42                         # ensures reproducibility
            )
        elif task == 'classification':
            # MLPClassifier: for discrete labels (e.g., cat vs dog)
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                random_state=42
            )
        else:
            # 🚨 If the user inputs an invalid task type, raise an error
            raise ValueError("Task must be 'classification' or 'regression'")

    # ------------------------------
    # 🚂 Train the model
    # ------------------------------
    def train(self, X, y):
        """
        Train the neural network on labeled data.

        Parameters:
        - X: array-like, shape [n_samples, n_features]
            The input data (e.g., feature vectors)
        - y: array-like, shape [n_samples]
            The labels (for classification) or target values (for regression)

        Training involves:
        - Forward pass: make predictions
        - Compute error: difference between predicted and true
        - Backward pass: adjust weights via backpropagation
        - Repeat over many iterations (max_iter)
        """
        self.model.fit(X, y)

    # ------------------------------
    # 🔮 Make predictions
    # ------------------------------
    def predict(self, X):
        """
        Use the trained model to make predictions on new/unseen data.

        Parameters:
        - X: array-like, shape [n_samples, n_features]
            The input features

        Returns:
        - Predicted values:
            - For classification: predicted class labels (e.g., 0, 1)
            - For regression: continuous numeric predictions (e.g., 42.3)
        """
        return self.model.predict(X)
