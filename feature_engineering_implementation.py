from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Loading the Telco processed Data
file_path = 'C:\\Users\\pillaiyar\\source\\repos\\dataanalysis_project\\preprocessed_telco_data.csv'
data = pd.read_csv(file_path)

# Feature Engineering

# a. Creating Interaction Term
# Example: Interaction between tenure and MonthlyCharges
data['tenure_monthlycharges_interaction'] = data['tenure'] * data['MonthlyCharges']

# b. Log Transformation
# Here we perform log transformation of TotalCharges to handle skewness
data['log_TotalCharges'] = np.log(data['TotalCharges'] + 1)

# c. Binning
# Hwe perform Binning tenure into groups
bins = [0, 12, 24, 36, 48, 60, 72]
labels = ['0-12', '12-24', '24-36', '36-48', '48-60', '60-72']
data['tenure_group'] = pd.cut(data['tenure'], bins=bins, labels=labels)

# d. Handling Missing Values
# Replacing NaN in TotalCharges with median
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)



# e. Scaling Features
# Scaling numerical features
scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                       'tenure_monthlycharges_interaction', 'log_TotalCharges']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Printing the Transformed Data
print(data.head())

# Setting style
sns.set(style="whitegrid")

# Creating a figure and a grid of subplots
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Plotting original TotalCharges distribution
sns.histplot(data['TotalCharges'], bins=30, kde=True, ax=ax[0, 0])
ax[0, 0].set_title('Original TotalCharges Distribution')

# Plotting log-transformed TotalCharges distribution
sns.histplot(data['log_TotalCharges'], bins=30, kde=True, ax=ax[0, 1])
ax[0, 1].set_title('Log-Transformed TotalCharges Distribution')

# Plotting original tenure distribution
sns.histplot(data['tenure'], bins=30, kde=True, ax=ax[1, 0])
ax[1, 0].set_title('Original Tenure Distribution')

# Plotting binned tenure distribution
sns.countplot(x='tenure_group', data=data, ax=ax[1, 1])
ax[1, 1].set_title('Binned Tenure Distribution')

# Saving the plots
plt.tight_layout()
plt.savefig('C:\\Users\\pillaiyar\\source\\repos\\dataanalysis_project\\images\\feature_distributions.png')
plt.show()

# Creating a figure and a grid of subplots
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plotting interaction feature
sns.scatterplot(x='tenure', y='MonthlyCharges', hue='tenure_monthlycharges_interaction', data=data, ax=ax[0])
ax[0].set_title('Interaction between Tenure and MonthlyCharges')

# Plotting binned tenure distribution
sns.countplot(x='tenure_group', data=data, ax=ax[1])
ax[1].set_title('Binned Tenure Distribution')

# Saving the plots
plt.tight_layout()
plt.savefig('C:\\Users\\pillaiyar\\source\\repos\\dataanalysis_project\\images\\distribution_tenure_monthlycharges_interaction.png.png')
plt.show()
