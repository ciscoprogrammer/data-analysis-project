import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'C:\\Users\\pillaiyar\\source\\repos\\dataanalysis_project\\WA_Fn-UseC_-Telco-Customer-Churn.csv'
telco_data = pd.read_csv(file_path)

# Identify Missing Values
missing_values = telco_data.isnull().sum()
print("Missing Values:")
print(missing_values)

# Ensure that data used for correlation is numeric
numeric_data = telco_data.select_dtypes(include=['float64', 'int64'])

# Correlation Matrix
correlation_matrix = numeric_data.corr()

# Visualize Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Variables')
# Save the figure
plt.savefig('C:\\Users\\pillaiyar\\source\\repos\\dataanalysis_project\\images\\correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()