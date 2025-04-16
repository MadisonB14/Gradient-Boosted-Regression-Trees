### This script reads in the results of the regression model, input parameters varied and the outbreak
### trait of interest calculated. Final result is a composed figure with 6 plots (5 box and whisker and 1 bar chart) 
### showing feature importance followed by box and whisker plots of the trends.

import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore
import matplotlib.patches as mpatches # type: ignore
from matplotlib.lines import Line2D # type: ignore

from sklearn.metrics import r2_score # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

plt.rcParams.update({'font.size': 14})

# Load the low movement models ONLY
model_1 = joblib.load('../1_model_file.joblib')
model_2 = joblib.load('../3_model_file.joblib')
model_3 = joblib.load('../5_model_file.joblib')

# Load the inputs
X1 = pd.read_csv('../../Movement_1_HostDensity_1.5.csv', header=0).values
X2 = pd.read_csv('../../Movement_1_HostDensity_3.csv', header=0).values
X3 = pd.read_csv('../../Movement_1_HostDensity_5.csv', header=0).values


# Load the outputs
y1 = pd.read_csv('../LowDens-LowMove-AreaAffected-at52.csv', header=None).values.ravel()
y2 = pd.read_csv('../MedDens-LowMove-AreaAffected-at52.csv', header=None).values.ravel()
y3 = pd.read_csv('../HighDens-LowMove-AreaAffected-at52.csv', header=None).values.ravel()


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
    'InfectPeriodLive': 'Infectious \n Period of \n Living (days)',
    'InfectPeriodCarcass': 'Infectious \n Period of \n Carcass (days)',
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

# Now plot the univariate plots for each feature
for i, feature in enumerate(input_labels):
    ax = axes[i + 1]  # Start from the second subplot (index 1)

    lightcolor = light_shade[i]
    medcolor = custom_colors1[i]  # Get the corresponding custom color for the feature
    highcolor = dark_shade[i]
    
    # Get the feature index (the column index of the feature in X_test)
    feature_idx = i

    # Define the values for the current feature in the test set
    feature_values = sorted(set(X1_test[:, feature_idx]))  # Get unique values for the feature

    # Prepare a list that will store each densities data
    boxdata = []
    positions = []
    widths = []

    for value in feature_values:
        print("Feature: ", feature)
        print("Value: ", value)

        # Create a mask for selecting the rows where the feature value matches
        low_mask = (X1_test[:, feature_idx] == value)
        medium_mask = (X2_test[:, feature_idx] == value)
        high_mask = (X3_test[:, feature_idx] == value)

        # Collect the model predictions for each density level for this feature value
        lowdens_predictions = model_1.predict(X1_test[low_mask])
        mediumdens_predictions = model_2.predict(X2_test[medium_mask])
        highdens_predictions = model_3.predict(X3_test[high_mask])        
        
        # Add the data for the three density levels to the box and whisker data list
        boxdata.append(lowdens_predictions)
        boxdata.append(mediumdens_predictions)
        boxdata.append(highdens_predictions)    

        # Add corresponding positions for each box in the cluster (low, medium, high)
        # Adjust positions based on the feature's requirements
        if i == 0:  # Example feature 1
            positions.extend([value - 1, value, value + 1])  # Slight spread
            widths.extend([0.4, 0.4, 0.4])
        elif i == 1:  # Example feature 2
            positions.extend([value - 1.5, value, value + 1.5])  # Wider spread
            widths.extend([0.9, 0.9, 0.9])
        elif i == 2:  # Example feature 3
            positions.extend([value - 0.8, value, value + 0.8])  # Default spread
            widths.extend([0.35, 0.35, 0.35])
        elif i == 3:  # Example feature 4
            positions.extend([value - 2.04, value - 1, value - 0.04])  # Custom spread
            widths.extend([0.04, 0.04, 0.04])
        else:
            positions.extend([value - 15, value, value + 15])  # Narrow spread for other features
            widths.extend([10, 10, 10])        

        # Add corresponding positions for each box in the cluster (low, medium, high)
        # This ensures that the boxes will be grouped correctly on the x-axis
        #positions.extend([value - 0.2, value, value + 0.2])  # Spread the positions slightly


    # Create the box plot for each feature: 3 boxes per feature value
    box = ax.boxplot(boxdata, showfliers=False, patch_artist=True, whis=(0, 100), positions=positions, widths=widths)

    # Define manual x-tick locations and labels for this feature (use your custom locations)
    # You need to adjust custom_ticks to match with the number of boxes
    custom_ticks = []
    custom_tick_labels = []

    # Manually set custom x-ticks and labels for each feature plot
    if i == 0:
        custom_ticks = [5, 10, 15, 20]  # Example: for InfectPeriodLive feature
        custom_tick_labels = ['5', '10', '15', '20']
    elif i == 1:
        custom_ticks = [5, 10, 15, 30, 50]  # Example: for InfectPeriodCarcass feature
        custom_tick_labels = ['5', '10', '15', '30', '50']
    elif i == 2:
        custom_ticks = [2, 4, 8, 12]  # Example: for IncubationPeriod feature
        custom_tick_labels = ['2', '4', '8', '12']
    elif i == 3:
        custom_ticks = [0.05, 0.1, 0.3, 0.5, 0.7]  # Example: for ProportionRecover feature
        custom_tick_labels = ['0.05', '   0.1', '0.3', '0.5', '0.7']
    elif i == 4:
        custom_ticks = [91, 182, 273, 364, 455, 546]  # Example: for DaysToRecovery feature
        custom_tick_labels = ['91', '182', '273', '364', '455', '546']

    ax.set_xticks(custom_ticks)  # Set x-tick positions
    ax.set_xticklabels(custom_tick_labels)  # Set x-tick labels and rotate them for clarity

    # Set labels and titles
    ax.set_xlabel(custom_x_labels[feature])  # Use custom x-axis labels from the dictionary

    # Check if the current plot is the first one in the bottom row
    if i == 0:  # top middle
        ax.set_ylabel('Area Affected at 52 Weeks (km$^2$)')
        ax.set_yticks([200, 400, 600, 800, 1000, 1200, 1400])  # Customize y-ticks for this plot
    elif i == 2:  # bottom left
        ax.set_ylabel('Area Affected at 52 Weeks (km$^2$)')
        ax.set_yticks([200, 400, 600, 800, 1000, 1200, 1400])  # Customize y-ticks for this plot

    # Hide y-axis ticks for all other plots
    if i != 0 and i != 2:
        ax.set_yticklabels([])  # Hide y-tick labels for other plots

        # Customize boxplot colors for the different density levels (low, medium, high)
    for i, patch in enumerate(box['boxes']):
        # Alternate the colors for the density levels (Low, Medium, High)
        if i % 3 == 0:  # Low density
            patch.set_facecolor(lightcolor)  # Color based on feature
        elif i % 3 == 1:  # Medium density
            patch.set_facecolor(medcolor)  # Define medium density color
        else:  # High density
            patch.set_facecolor(highcolor)  # Define high density color
    
    for median_line in box['medians']:
        median_line.set_color('black')
        median_line.set_linewidth(2)

######################################################################
#### Redo Proportion that recover plot because she's challenging #####
######################################################################

ax = axes[4]
ax.clear()
index=3
boxdata1 = []
position = []
propthatrecov = sorted(set(X1_test[:, 3]))  # Get unique values for the feature

custom_tick_labels = [str(value) for value in propthatrecov]  # Convert the values to strings for labels

group_spacing = 0.8  # Adjust this spacing as needed for your visualization


for idx, value in enumerate(propthatrecov):
    base_position = idx * group_spacing  # This will create enough space between groups


    # Create masks for selecting the rows where the feature value matches (for the 4th feature)
    low_mask = (X1_test[:, index] == value)
    medium_mask = (X2_test[:, index] == value)
    high_mask = (X3_test[:, index] == value)

    # Collect the model predictions for each density level for this feature value
    lowdens_predictions = model_1.predict(X1_test[low_mask])
    mediumdens_predictions = model_2.predict(X2_test[medium_mask])
    highdens_predictions = model_3.predict(X3_test[high_mask])        
        
    # Add the data for the three density levels to the box and whisker data list
    boxdata1.append(lowdens_predictions)
    boxdata1.append(mediumdens_predictions)
    boxdata1.append(highdens_predictions)

    # Plot the new boxplot
    position.extend([base_position - 0.15, base_position, base_position + 0.15])  # Spread the boxes within the group

box1 = ax.boxplot(boxdata1, showfliers=False, patch_artist=True, whis=(0, 100), positions=position, widths=0.08)

# Set custom x-ticks based on the unique values in propthatrecov
# We need to select only the x-tick positions corresponding to the feature values
# For example, we'll select positions that fall within the range of the feature values
tick_positions = [idx * group_spacing for idx in range(len(propthatrecov))]  # Set x-ticks based on the index of the values
tick_labels = custom_tick_labels  # Labels should correspond to propthatrecov values

# Set the x-ticks and labels
ax.set_xticks(tick_positions)  # Set the ticks where the feature values are located
ax.set_xticklabels(tick_labels)  # Set labels for these ticks
ax.set_xlabel('Proportion that Recover')
ax.set_yticklabels([])

for median_line in box1['medians']:
    median_line.set_color('black')
    median_line.set_linewidth(2)

for i, patch in enumerate(box1['boxes']):
    # Alternate the colors for the density levels (Low, Medium, High)
    if i % 3 == 0:  # Low density
        patch.set_facecolor('#f8e28b')  # Color based on feature
    elif i % 3 == 1:  # Medium density
        patch.set_facecolor('#F4CF3E')  # Define medium density color
    else:  # High density
        patch.set_facecolor('#927c25')  # Define high density color

#########################################################################

# Remove any extra subplots (if we have fewer than 8 features)
for j in range(i + 2, len(axes)):  # Starting from the first empty subplot
    fig.delaxes(axes[j])

# Adjust layout and save the figure
fig.tight_layout()
plt.savefig('BoxWhisk-AreaAffected.png', dpi=300)
# plt.show()
