"""
Normal Equation approach to Linear Regression yourself, implement it and develop a regression model. Evaluate the model using the evaluation metrics
"""
#Importing the modules
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# Loading the data
df = pd.read_csv("C:\\Users\\HP\\SML\\Sample data\\Real estate.csv")

"""
Extract input features (X) and output variable (y)
:, 1:-1 df.iloc.values gives a numpy array with the values of the chosen columns, which make up the dataset's input features. 
The X variable is given this numpy collection.
:, -1 in df.iloc.values only chooses the output variable's last column in the DataFrame. 
The y variable has been given the numpy collection.
Using NumPy's c_ function, to add a column of ones to the X matrix to add a bias component to the input features.
"""
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
X = np.c_[np.ones(X.shape[0]), X]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
Computing the parameter vector theta of a linear regression model using the normal equation.
first transpose the training data (X_train.T), then increase it by X_train , To find the product X_train.T.dot(X_train), . 
Then, using np.linalg.inv(), we take the inverse of this result and multiply it by the transposition of the training data, X_train.T. As a result, we get the value np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T). 
The parameter vector theta is then obtained by multiplying this by the training labels y_train.
"""
theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

# prediction on testing set
y_pred = X_test.dot(theta)

"""
Evaluating the model using the evaluation metrics
"""
"""
mean_squared_error function calculates the mean squared error (MSE) between the true values y_true and the predicted values y_pred. The MSE is computed by taking the average of the squared differences between the predicted and actual values. 
It gives an idea about how far off our predicted values are from the actual values.
"""
mse = np.mean((y_pred - y_test)**2)
def root_mean_squared_error(y_true, y_pred):
    return np.mean((y_pred - y_true)**2)
"""
The mean absolute error (MAE) between the actual and predicted values is determined by the mean_absolute_error function. 
The average of the absolute differences between the predicted and real values is used to calculate the MAE. 
"""
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))
"""
The R-squared (coefficient of determination) value between the actual and predicted values is determined by the r2_score function. 
The mathematical indicator of how closely the data resemble the fitted regression line is called R-squared. 
It has a scale of 0 to 1, with 1 denoting a perfect match between the model and the data. The entire sum of squared differences between actual and mean values makes up the denominator, while the numerator is the sum of squared differences between predicted and actual values. 
The better the model fits the data, the closer the R-squared number is to 1.
"""
def r2_score(y_true, y_pred):
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (numerator / denominator)


print('Mean Squared Error:', mse)
# Compute RMSE
rmse = np.sqrt(root_mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', rmse)
# Compute MAE
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)
# Compute R-squared
r2 = r2_score(y_test, y_pred)
print('R-squared:', r2)
