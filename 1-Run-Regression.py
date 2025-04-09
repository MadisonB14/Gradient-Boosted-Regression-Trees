import matplotlib.pyplot as plt
import numpy as np # type: ignore
import pandas as pd
import joblib

from sklearn import ensemble # type: ignore
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

## File preparation 
inputs = pd.read_csv('../Movement_1_HostDensity_1.5.csv', header=0)
outputs = pd.read_csv('LowDens-LowMove-AreaAffected-at52.csv', header=None).values.ravel()

## Training and Testing the Model

# Split up the data: training or testing
# random_state = 35 to make this reproducible
# test_size = % of data that will be used for testing, the rest to be trained on 80/20
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, 
                                                    test_size=0.2, random_state=35)

X = inputs
y = outputs

params = {
    "n_estimators": 1000,
    "max_depth": 8,
    "min_samples_split": 6,
    "learning_rate": 0.005,
    "loss": "squared_error",
}

# Train the model!
reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

# Test the model on real data and evaluate
mse = mean_squared_error(y_test, reg.predict(X_test))

with open('1-regression-results.out', 'w') as file:
    # Writing the formatted strings to the file
    file.write("The mean squared error (MSE) on test set: {:.4f}\n".format(mse))
    file.write("R-squared [score function] (training): {0:.3f}\n".format(reg.score(X_train, y_train)))
    file.write("R-squared [score function] (validation): {0:.3f}\n".format(reg.score(X_test, y_test)))

joblib.dump(reg, '1_model_file.joblib')



