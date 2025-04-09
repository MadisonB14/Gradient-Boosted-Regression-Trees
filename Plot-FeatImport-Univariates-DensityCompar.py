import matplotlib.pyplot as plt
import numpy as np # type: ignore
import pandas as pd
import joblib
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load the low movement models ONLY
model_1 = joblib.load('1_model_file.joblib')
model_2 = joblib.load('3_model_file.joblib')
model_3 = joblib.load('5_model_file.joblib')

# Load the inputs
X1 = pd.read_csv('../Movement_1_HostDensity_1.5.csv', header=0).values
X2 = pd.read_csv('../Movement_1_HostDensity_3.csv', header=0).values
X3 = pd.read_csv('../Movement_1_HostDensity_5.csv', header=0).values


# Load the outputs
y1 = pd.read_csv('LowDens-LowMove-AreaAffected-at52.csv', header=None).values.ravel()
y2 = pd.read_csv('MedDens-LowMove-AreaAffected-at52.csv', header=None).values.ravel()
y3 = pd.read_csv('HighDens-LowMove-AreaAffected-at52.csv', header=None).values.ravel()


# Split the data for testing and training
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=35)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=35)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=35)

# Train the models
model_1.fit(X1_train, y1_train)
model_2.fit(X2_train, y2_train)
model_3.fit(X3_train, y3_train)

# Make predictions for both models on their test data
y1_pred = model_1.predict(X1_test)
y2_pred = model_2.predict(X2_test)
y3_pred = model_3.predict(X3_test)

# Evaluate both models using R² score
r2_1 = r2_score(y1_test, y1_pred)
r2_2 = r2_score(y2_test, y2_pred)
r2_3 = r2_score(y3_test, y3_pred)

print(f"R² for Model 1: {r2_1}")
print(f"R² for Model 2: {r2_2}")
print(f"R² for Model 3: {r2_3}")

########################################################
##### Plotting Feature Importance with Univariates #####
########################################################
input_labels = ['InfectPeriodLive', 'InfectPeriodCarcass', 
               'IncubationPeriod','ProportionRecover', 'DaysToRecovery']

custom_colors1 = [
    '#5AB1BB',  # moonstone Infect Per Liv
    '#B948B7',  # purple Incub Per
    '#A5C882',  # pistachio Infect Per Car
    '#F4CF3E',   # yellow Prop that Recover
    '#EF933B'   # orange Recovery
]

light_shade = [
    '#add8dd',
    '#d591d4',
    '#c9deb4',
    '#f8e28b',
    '#f5be89'

]

dark_shade = [
    '#2d595e',
    '#823280',
    '#63784e',
    '#927c25',
    '#a76729'
]

custom_x_labels_feat = { # for feat importance
    'InfectPeriodLive': 'Infectious Period \n of Living (days)',
    'InfectPeriodCarcass': 'Infectious Period \n of Carcass (days)',
    'IncubationPeriod': 'Incubation \n Period (days)',
    'ProportionRecover': 'Proportion \n that Recover',
    'DaysToRecovery': 'Recovery \n Period (days)'
}

custom_x_labels = { # for univariate
    'InfectPeriodLive': 'Infectious Period of Living (days)',
    'InfectPeriodCarcass': 'Infectious Period of Carcass (days)',
    'IncubationPeriod': 'Incubation Period (days)',
    'ProportionRecover': 'Proportion that Recover',
    'DaysToRecovery': 'Recovery Period (days)'
}

feature_importances_1 = model_1.feature_importances_
feature_importances_2 = model_2.feature_importances_
feature_importances_3 = model_3.feature_importances_

# Create a 2x4 grid for subplots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))  # Adjust size to fit all plots
axes = axes.flatten()  # Flatten the axes array to make indexing easier

# Plot the Feature Importance in the first subplot (top-left)
ax = axes[0]
bar_width = 0.30  # Width of the bars
index = np.arange(len(input_labels))

# # Plot the bars for Model 1 and Model 2 side by side
for i in range(len(input_labels)):
    ax.barh(index[i] - bar_width, feature_importances_1[i], bar_width, align="center", color=light_shade[i], edgecolor='black', label='Low')
    ax.barh(index[i], feature_importances_2[i], bar_width, align="center", color=custom_colors1[i], edgecolor='black', label='Med')
    ax.barh(index[i] + bar_width, feature_importances_3[i], bar_width, align="center", color=dark_shade[i], edgecolor='black', label='High')


# Customize the feature importance plot
ax.set_yticks(index)
ax.set_yticklabels([custom_x_labels_feat[label] for label in input_labels])  # Use custom feature names
ax.set_xlabel("Feature Importance (MDI)")
# ax.legend()

# Create white-colored proxy artists for the legend
proxy_model_1 = mpatches.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label=f'Low Density (R²: {r2_1:.2f})')
proxy_model_2 = mpatches.Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='black', label=f'Med. Density (R²: {r2_2:.2f})')
proxy_model_3 = mpatches.Rectangle((0, 0), 1, 1, facecolor='darkgray', edgecolor='black', label=f'High Density (R²: {r2_3:.2f})')
ax.legend(handles=[proxy_model_3, proxy_model_2, proxy_model_1], loc='upper right')

offset = [0.45, 1.5, 0.35, 0.02, 15]

# Now plot the univariate plots for each feature
for i, feature in enumerate(input_labels):
    ax = axes[i + 1]  # Start from the second subplot (index 1)
    color = custom_colors1[i]  # Get the corresponding custom color for the feature
    
    # Get the feature index (the column index of the feature in X_test)
    feature_idx = i

    # Scatter plot for Model 1 vs predicted output
    ax.scatter(X1_test[:, feature_idx] - offset[i] , model_1.predict(X1_test), color=light_shade[i], alpha=0.7, s=10, label="Low")

    # Scatter plot for Model 2 vs predicted output
    ax.scatter(X2_test[:, feature_idx], model_2.predict(X2_test), color=color, alpha=0.7, s=10, label="Med")

    # Scatter plot for Model 3 vs predicted output
    ax.scatter(X3_test[:, feature_idx] + offset[i], model_3.predict(X3_test), color=dark_shade[i], alpha=0.7, s=10, label="High")
    
    # Define manual x-tick locations and labels for this feature (use your custom locations)
    if feature == 'InfectPeriodLive':
        custom_ticks = [5, 10, 15, 20]  # Example: for InfectPeriodLive feature
        custom_tick_labels = ['5', '10', '15', '20']
    elif feature == 'InfectPeriodCarcass':
        custom_ticks = [5, 10, 15, 30, 50]  # Example: for InfectPeriodCarcass feature
        custom_tick_labels = ['5', '10', '15', '30', '50']
    elif feature == 'IncubationPeriod':
        custom_ticks = [2, 4, 8, 12]  # Example: for IncubationPeriod feature
        custom_tick_labels = ['2', '4', '8', '12']
    elif feature == 'ProportionRecover':
        custom_ticks = [0.05, 0.1, 0.3, 0.5, 0.7]  # Example: for ProportionRecover feature
        custom_tick_labels = ['0.05', '   0.1', '0.3', '0.5', '0.7']
    elif feature == 'DaysToRecovery':
        custom_ticks = [91, 182, 273, 364, 455, 546]  # Example: for DaysToRecovery feature
        custom_tick_labels = ['91', '182', '273', '364', '455', '546']
    
    # Apply custom x-tick locations and labels
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_tick_labels)

    # Set labels and titles
    ax.set_xlabel(custom_x_labels[feature])  # Use custom x-axis labels from the dictionary
    # ax.set_ylabel('Predicted Output')
    # ax.set_title(f'{feature} vs Predicted Output')

    num_cols = 3  # Number of columns in the subplot grid (adjust this based on your layout)
    total_plots = len(axes)
    
    # Check if the current plot is the first one in the bottom row
    if i == total_plots - num_cols - 1:  # The first plot in the bottom row (e.g., index 6 in a 3x3 grid)
        ax.set_ylabel('Area Affected at 52 Weeks (km$^2$)')    

# Remove any extra subplots (if we have fewer than 8 features)
for j in range(i + 2, len(axes)):  # Starting from the first empty subplot
    fig.delaxes(axes[j])

# Adjust layout and save the figure
# fig.suptitle('Area Affected at 52 Weeks')
fig.tight_layout()
plt.savefig('allshaded-AllDens-areaaffected-at52.png', dpi=300)
# plt.show()
