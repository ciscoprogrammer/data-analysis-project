import pandas as pd

from sklearn.model_selection import train_test_split

# Load the dataset
file_path = 'C:\\Users\\pillaiyar\\source\\repos\\dataanalysis_project\\WA_Fn-UseC_-Telco-Customer-Churn.csv'
telco_data = pd.read_csv(file_path)
telco_data['TotalCharges'] = pd.to_numeric(telco_data['TotalCharges'], errors='coerce')

# Handle NaN values: replacing NaN in 'TotalCharges' with its mean value
mean_value = telco_data['TotalCharges'].mean()
telco_data['TotalCharges'].fillna(mean_value, inplace=True)
# encode categorical variables
telco_data_encoded = pd.get_dummies(telco_data, columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'], drop_first=True)
print(telco_data_encoded.columns)

# Defining Feature and target value
X = telco_data_encoded.drop(columns=['customerID', 'Churn_Yes'])
y = telco_data_encoded['Churn_Yes']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#export to csv:
telco_data_encoded.to_csv('C:\\Users\\pillaiyar\\source\\repos\\dataanalysis_project\\preprocessed_telco_data.csv', index=False)
