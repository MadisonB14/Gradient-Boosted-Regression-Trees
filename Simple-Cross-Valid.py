import numpy as np # type: ignore
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

## File preparation 
X = pd.read_csv('grouped_Movement_1_HostDensity_1.5.csv', header=0)  # Keep as DataFrame
y = pd.read_csv('../LowDens-LowMove-AreaAffected-at52.csv', header=None).values.ravel()  # Use .ravel() to flatten into 1D array

params = {
    "n_estimators": 1000,
    "max_depth": 8,
    "min_samples_split": 6,
    "learning_rate": 0.08,
    "loss": "squared_error",
}

groups = X['System_ID']

# Set up GroupKFold
group_kfold = GroupKFold(n_splits=5)

# Initialize the model
model = GradientBoostingRegressor(**params)

# For storing results
r2_scores = []

# Cross-validation loop
for train_idx, val_idx in group_kfold.split(X, y, groups):
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
    
    # Train the model
    model.fit(X_train_cv, y_train_cv)
    
    # Predict on the validation set
    y_pred_cv = model.predict(X_val_cv)
    
    # Evaluate using R-squared (R²)
    r2 = r2_score(y_val_cv, y_pred_cv)
    r2_scores.append(r2)


# Open a file in write mode
with open('simpleCV-results.out', 'w') as file:
    # Writing the information to the file
    file.write(f"Average R squared: {np.mean(r2_scores):.4f}")
##########################################################
