# Customer-Churn-prediction
Customer churn prediction
Problem Description :
Customer churn is the term used to describe the loss of customers from a business or service. It is an important metric for businesses to monitor as it can have a significant impact on revenue and profitability. In this project, we will be predicting customer churn for a telecommunications company based on various customer attributes and their usage patterns.
Dataset Description:
The dataset used for this project contains information about customers of a telecommunications company. It includes demographic information such as age, gender, and income, as well as information about the customer’s account such as contract type, payment method, and monthly charges. Additionally, it includes information about the customer’s usage of the company’s services such as the number of phone lines, internet usage, and type of service. The dataset also includes a binary variable indicating whether the customer has churned or not.
Project Objective:
The objective of this project is to predict whether a customer is likely to churn or not based on their demographic and usage patterns. By identifying customers who are at high risk of churning, the company can take proactive measures to retain them and reduce churn rates.
Framework and Steps:
1. Data Collection: Collect the dataset from the data source.
2. Data Preprocessing: Preprocess the data by handling missing values, removing duplicates, encoding categorical variables, and scaling numerical variables.
3. Exploratory Data Analysis (EDA): Perform EDA to understand the relationship between the target variable and the features, and identify any patterns or trends in the data.
4. Feature Selection: Select the relevant features for the prediction model using feature selection techniques.
5. Model Selection: Select the appropriate machine learning model based on the problem type and data characteristics.
6. Model Training: Train the selected model on the preprocessed data.
7. Model Evaluation: Evaluate the performance of the model using appropriate evaluation metrics.
8. Hyperparameter Tuning: Tune the hyperparameters of the selected model to improve its performance.
9. Prediction: Use the trained model to make predictions on new data.
Code Explanation :
Here is the simple explanation for the code which is provided in the code.py file.
The code is for customer churn prediction using a dataset. The dataset contains information about customers and their churn status. We first load the dataset and perform data exploration, checking for missing values, data types, and correlations between features.
Next, we preprocess the data by encoding categorical features, scaling numerical features, and splitting the data into training and testing sets.
Then, we train several classification models, including Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine. We evaluate the models using accuracy, precision, recall, and F1-score metrics.
Finally, we select the best performing model based on evaluation metrics and make predictions on new data to predict whether a customer is likely to churn or not.
How to Run the Code
To run the code, follow the steps below:
1. Download the dataset from the source mentioned in the code.
2. Install the required libraries listed at the top of the code, such as pandas, numpy, scikit-learn, and matplotlib.
3. Open the Python environment or Jupyter Notebook and navigate to the directory where the code is saved.
4. Run each section of the code in order, making sure to install any missing libraries as needed.
5. After training and evaluating the models, select the best performing model based on evaluation metrics and use it to make predictions on new data.
Requirements to Run the Code
The code requires the following libraries to be installed:
• Pandas
• Numpy
• Scikit-learn
• Matplotlib

