# Import the XGBoost library
# XGBoost stands for "Extreme Gradient Boosting"
# It's a powerful machine learning algorithm based on the Gradient Boosting framework
import xgboost as xgb


class XGBoostModel:
    def __init__(self, task="regression"):
        """
        task: either "regression" or "classification"
        """

        self.task = task

        # GRADIENT BOOSTING PRIMER 🌱
        # --------------------------
        # Gradient Boosting builds an ensemble of decision trees in a sequential manner.
        # Each new tree is trained to correct the errors (residuals) made by the previous trees.
        # Think of it like this: Model_1 makes a prediction. Model_2 tries to fix the mistakes of Model_1.
        # This continues for many rounds (n_estimators times), and the models "boost" each other!

        # Why is it "gradient"? Because the algorithm uses gradient descent to minimize error.
        # At each step, it fits a new tree to the negative gradient of the loss function (i.e., it follows the error).

        # XGBOOST OVERVIEW 🚀
        # -------------------
        # XGBoost is an optimized, scalable version of Gradient Boosting.
        # It's faster and more accurate due to:
        # - Parallel processing
        # - Regularization (to avoid overfitting)
        # - Pruning and handling of missing values
        # - Built-in support for early stopping, cross-validation, and more

        # Model setup for regression tasks
        if task == "regression":
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',   # Regression loss function: squared error
                n_estimators=100,               # Number of boosting rounds (trees)
                learning_rate=0.1,              # Step size shrinkage; lower = more accurate but slower
                max_depth=3,                    # Maximum depth of each decision tree (controls complexity)
                random_state=42                 # Ensures reproducibility
            )

        # Model setup for classification tasks
        elif task == "classification":
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',    # For binary classification problems
                n_estimators=100,               # Number of boosting rounds
                learning_rate=0.1,              # Smaller = more boosting rounds needed, but better generalization
                max_depth=3,                    # Controls the depth/complexity of each tree
                use_label_encoder=False,        # Avoids deprecated label encoder warning
                eval_metric='logloss',          # Common classification metric: lower is better
                random_state=42                 # Ensures reproducible results
            )

        else:
            raise ValueError("Invalid task type. Use 'regression' or 'classification'.")

    def train(self, X, y):
        # Fit the model to the data (training phase)
        # X: features, y: target labels
        # XGBoost uses boosting logic under the hood when training
        self.model.fit(X, y.ravel())

    def predict(self, X):
        # Make predictions using the trained model
        return self.model.predict(X)
