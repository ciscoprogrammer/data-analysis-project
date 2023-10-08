
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Loading the preprocessed Data
file_path = 'C:\\Users\\pillaiyar\\source\\repos\\dataanalysis_project\\preprocessed_telco_data.csv'
telco_data = pd.read_csv(file_path)


# Defining X and y
X = telco_data.drop(columns=['customerID', 'TotalCharges'])
y = telco_data['TotalCharges']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializeing the model
linreg = LinearRegression()

# Training the model
linreg.fit(X_train, y_train)

# Predictions
y_pred = linreg.predict(X_test)

# Evaluation 
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Visualizating Model Predictions vs Actual Values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual TotalCharges")
plt.ylabel("Predicted TotalCharges")
plt.title("Actual vs. Predicted TotalCharges")
plt.savefig('C:\\Users\\pillaiyar\\source\\repos\\dataanalysis_project\\linearreg_actual_vs_predicted_totalcharges.png', dpi=300, bbox_inches='tight')

plt.show()

# Feature Importance
feature_importance = pd.Series(linreg.coef_, index=X_train.columns)
feature_importance = feature_importance.abs().sort_values(ascending=False)
feature_importance.plot(kind='bar', figsize=(12,6))
plt.title('Feature Importance (Linear Regression Coefficients)')
plt.savefig('C:\\Users\\pillaiyar\\source\\repos\\dataanalysis_project\\linearreg_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()



