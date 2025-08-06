# Wrapper for Random Forest Regressor 🌲📦

from sklearn.ensemble import RandomForestRegressor

"""
🌲 RANDOM FOREST BEHAVIOR EXPLAINED — ALL IN HYPERPARAMETERS! 🌲

Random Forests are powerful because they combine:
1. Bagging (bootstrap sampling of data)
2. Feature randomization (random feature subsets for splitting)
3. Ensemble learning (many weak trees → strong model)

✅ You can simulate most variations of Random Forests by tuning hyperparameters:

---------------------------------------------------------
🧺 1. BAGGING (Bootstrap Aggregation)
---------------------------------------------------------
- Each tree is trained on a random subset of the training data (with replacement).
- Enabled by: `bootstrap=True` (default in sklearn)
- This promotes diversity and reduces overfitting.

---------------------------------------------------------
🧠 2. FEATURE RANDOMIZATION (a.k.a. Random Subspaces)
---------------------------------------------------------
- At each split, the model randomly selects a subset of features.
- Controlled by the `max_features` parameter:
    - `sqrt` → Square root of total features (default for regression)
    - `log2` → Log base 2 of total features
    - Float (e.g., 0.5) → Use % of features
    - `None` → Use all features (turns off feature randomization)

---------------------------------------------------------
🌲 3. TREE DEPTH AND SIZE
---------------------------------------------------------
- Control how deep or large trees grow (and how complex they get):
    - `max_depth` → Maximum depth of any tree
    - `min_samples_split` → Min samples to split a node
    - `min_samples_leaf` → Min samples per leaf node

---------------------------------------------------------
🔄 4. MODEL DIVERSITY
---------------------------------------------------------
- Use different `random_state` values to train varied forests.
- You can train multiple forests on different data slices.

---------------------------------------------------------
🌟 5. SPECIAL VARIANTS & BEYOND
---------------------------------------------------------
- **ExtraTrees**: Adds even more randomness (use `ExtraTreesRegressor`)
- **Boosted Trees**: Sequential trees instead of parallel (use `GradientBoostingRegressor`)
- **Random Subspace Forests**: Random features per tree, not just per split — requires custom implementation

TL;DR 🧠:
✅ You’re already using Bagging & Feature Randomization by default!
🎛️ Use hyperparameters to control model complexity, diversity, and behavior.

"""

# 🔧 Common Hyperparameters You Can Customize in RandomForestRegressor:

# n_estimators      : Number of trees in the forest (higher = more accurate, slower)
# max_depth         : Maximum depth of each tree (limits tree growth, prevents overfitting)
# min_samples_split : Minimum number of samples needed to split a node (higher = more conservative)
# min_samples_leaf  : Minimum number of samples at a leaf node (prevents overgrowth of trees)
# max_features      : Number of features considered when splitting (can be 'auto', 'sqrt', or a number)
# bootstrap         : Whether bootstrap samples are used (True = default and recommended)
# n_jobs            : Number of CPU cores to use (-1 = use all cores for speed)

# ✅ Tips:
# - Use GridSearchCV to automatically test multiple hyperparameter combinations.
# - Set n_jobs=-1 to use all cores for faster training.
# - Shallow trees generalize better; deep trees may overfit.

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        """
        🔧 Constructor method that initializes the RandomForestRegressor.
        
        Args:
            n_estimators (int): Number of trees in the forest (more = better, but slower).
            random_state (int): Seed value to ensure reproducible results (same output each run).
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,       # 🌲 Number of decision trees in the ensemble
            random_state=random_state        # 🔁 Random seed to ensure same results every time
        )


    def train(self, X, y):
        """
        Fit the model to training data.
        Args:
            X: Input features (2D array)
            y: Target values (1D or 2D array)
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict using the trained model.
        Args:
            X: Input features for prediction
        Returns:
            Predicted values
        """
        return self.model.predict(X)
