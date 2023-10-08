# data-analysis-project
# Customer Churn Prediction & Customer Lifetime Value Prediction



## Project Overview
This project involves the application of various data analysis techniques and model development to predict customer churn and Customer Lifetime Value (CLV) based on a dataset from a telecommunications company. The dataset, `WA_Fn-UseC_-Telco-Customer-Churn.csv`, contains customer information, service usage, and churn data.



### Objectives

1. **Exploratory Data Analysis (EDA)**

   - Initial exploration of the dataset to understand the distributions and relationships between different variables.
   

2. **Data Preprocessing**

   - Handling missing values, encoding categorical variables, and scaling features to prepare the data for modeling.


3. **Classification Task**

   - Building a classification model to predict customer churn using various algorithms and evaluating model performance.


4. **Regression Task**

   - Constructing a regression model to predict CLV, evaluating, and interpreting its performance and results.


5. **Feature Engineering**

   - Enhancing the model by creating additional features, explaining the rationale behind each.



## Implementation

### Exploratory Data Analysis


- Loaded the dataset and performed an initial exploration to understand the underlying data.
- Visualized distributions and explored relationships between different variables.
- Handled missing values and encoded categorical variables.



### Data Preprocessing


- Handled missing data, one-hot encoded categorical variables, and split the data into training and test sets.



### Classification Task

- Built and trained a classification model using algorithms like logistic regression and decision trees to predict customer churn.
- Evaluated model performance using metrics like accuracy, precision, recall, F1-score, and ROC AUC.



### Regression Task

- Developed a regression model to predict Customer Lifetime Value (CLV).
- Evaluated the model using regression metrics like RMSE and R-squared.



### Feature Engineering

- Created interaction features, performed log transformations, binned numerical features, handled outliers, and scaled numerical variables to improve model performance.



## Technology Stack

- **Python**: The project is implemented using Python.
- **Pandas**: Used for data manipulation and analysis.
- **NumPy**: Used for numerical operations.
- **Scikit-learn**: Used for building machine learning models.
- **Matplotlib and Seaborn**: Used for visualizing data and results.



## How to Use

1. **Data Preprocessing**: Run `data_preprocessing.py` for initial data cleaning and preprocessing.

2. **Exploratory Data Analysis**: Use `telco_churn_eda.py` to understand the underlying patterns in the data.

3. **Feature Engineering**: Execute `feature_engineering_implementation.py` to generate new features.

4. **Model Training and Evaluation**: Run `model_fitting_and_evaluation.py` for training and evaluating models.

5. **Regression Task**: Execute `regression_task.py` to predict the Customer Lifetime Value (CLV).





## Acknowledgements
- Dataset Source: [Kaggle](Link to the dataset if available)

