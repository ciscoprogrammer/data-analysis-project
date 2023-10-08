import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
# Loading preprocessed data
file_path = 'C:\\Users\\pillaiyar\\source\\repos\\dataanalysis_project\\preprocessed_telco_data.csv'
telco_data = pd.read_csv(file_path)
print(telco_data.columns)

# Check for NaN or Inf values
print("NaN values in dataset:", telco_data.isna().sum().sum())
print("Inf values in dataset:", (telco_data == np.inf).sum().sum())




# Defining feature matrix and target variable
X = telco_data.drop(columns=['customerID', 'Churn_Yes'])  
y = telco_data['Churn_Yes']  

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 1. Model Building: Logistic Regression
logreg = LogisticRegression(random_state=42)

# 2. Model Training
logreg.fit(X_train, y_train)

# 3. Model Evaluation
# Predictions
y_pred = logreg.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 4. Visualizing Model Performance
# ROC Curve
y_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('C:\\Users\\pillaiyar\\source\\repos\\dataanalysis_project\\images\\Roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('C:\\Users\\pillaiyar\\source\\repos\\dataanalysis_project\\images\\precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Feature Importance
# (Logistic Regression Coefficients as Importance)
feature_importance = pd.Series(logreg.coef_[0], index=X_train.columns)
feature_importance = feature_importance.abs().sort_values(ascending=False)
feature_importance.plot(kind='bar', figsize=(12,6))
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.savefig('C:\\Users\\pillaiyar\\source\\repos\\dataanalysis_project\\images\\confusion_matrix_logreg.png', dpi=300, bbox_inches='tight')

plt.show()

