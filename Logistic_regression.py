"""
Implementing Logistic Regression from scratch and perform binary classification
"""

# Importing the modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Loading the data
data = pd.read_csv("C:\\Users\\HP\\SML\\Sample data\\Heart.csv")
# Converting categorical variables to binary columns
cat_cols = ['Sex', 'Fbs', 'RestECG', 'ExAng', 'Slope', 'Ca', 'Thal']
data = pd.get_dummies(data, columns=cat_cols)

class LogisticRegression:
    """
    initializing the model's hyperparameters. 
    The learning rate indicates how frequently the model changes its parameters while being trained, and the num_iterations shows the number of times the model will go through the training data.
    """
    def __init__(self, alpha=0.01, itr=1000):
        self.alpha = alpha
        self.itr = itr

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    """
    Using the inputs X and Y, this technique trains the logistic regression model. 
    A column of ones is first added to X to represent the intercept word. 
    The model values (theta) are then set to zeros. The technique uses the sigmoid function of the linear combination of the input features and the model parameters to compute the predicted probability (h) during training. (theta). 
    Then, using the gradient descent algorithm and the predetermined learning rate, it determines the gradient of the loss function (cross-entropy loss) with regard to the model parameters. This procedure is done as many times as necessary.
    """
    def fit(self, X, y):
        # adding a column of ones to X for the intercept term
        X = np.insert(X, 0, 1, axis=1)
        # initializing theta to zeros
        self.theta = np.zeros(X.shape[1])
        for i in range(self.itr):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.alpha * gradient
    
    """
    Using fresh data X, this technique creates predictions. The training data's mean and standard deviation are used to adjust X in the beginning (using StandardScaler() from scikit-learn). 
    The intercept word is then added to X by adding a column of ones. 
    Using the sigmoid function of the linear combination of the input features and the model parameters (theta) acquired during training, it determines the predicted probability of each sample. To obtain the expected class label, the predicted probability is then rounded to 0 or 1.
    """
    def predict(self, X):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        # adding a column of ones to X for the intercept term
        X = np.insert(X, 0, 1, axis=1)
        y_pred = self.sigmoid(np.dot(X, self.theta))
        return np.round(y_pred)
    
    """
    With the help of a set of true labels (y_true) and predicted labels (y_pred), this technique determines the model's accuracy. 
    It provides the percentage of samples in the dataset that were properly classified.
    """
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

"""
Extracting the features and target variables
The features, denoted by X, were obtained by removing two columns from the original dataset—namely, "AHD" and "ChestPain"—using the Pandas DataFrame's drop technique.
The target variable, denoted by y, is created by using the replace technique of the Pandas Series to swap out the values in the 'AHD' column for binary values of 1 and 0. 
The values "Yes" are changed to 1 and the values "No" to 0 respectively.
"""
X = data.drop(['AHD', 'ChestPain'], axis=1)
y = data['AHD'].replace({'Yes': 1, 'No': 0})

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""
LogisticRegression object is created and the fit() method is called to train the model on the training data.
"""
lr = LogisticRegression()
lr.fit(X_train, y_train)

"""
To generate predictions based on the trained model, the predict() method is used on the testing data. 
Finally, using the accuracy() method of the LogisticRegression class, the model's accuracy is assessed by contrasting the predicted values with the real values.
"""
y_pred = lr.predict(X_test)
acc = lr.accuracy(y_test, y_pred)
print(f"Accuracy: {acc}")
