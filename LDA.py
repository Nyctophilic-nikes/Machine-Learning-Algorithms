"""
Implementing LDA from scratch on iris dataset
"""

# Importing modules
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Loading the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

class LDA:
    def __init__(self, n_components=None):
        self.n_components = n_components
    """
    The training data X and the labels that go with them are passed to the fit() function. 
    The data are divided into classes, the mean vectors for each class are computed, and finally the within-class and between-class scatter matrices are computed. 
    Then, if n_components is given, it calculates the eigenvectors and eigenvalues of the matrix (within_class_scatter)(-1)*between_class_scatter), arranges them in ascending order, and chooses the top n_components eigenvectors. 
    The projection matrix used for dimensionality reduction is made up of these eigenvectors.
    """
    def fit(self, X, y):
        separated = {}
        for i in range(len(X)):
            x = X[i]
            label = y[i]
            if label not in separated:
                separated[label] = []
            separated[label].append(x)
        mean_vectors = {}
        for label, data in separated.items():
            mean_vectors[label] = np.mean(data, axis=0)
        within_class_scatter = np.zeros((X.shape[1], X.shape[1]))
        for label, data in separated.items():
            scatter_matrix = np.zeros((X.shape[1], X.shape[1]))
            for row in data:
                row, mean_vectors[label]
                scatter_matrix += (row - mean_vectors[label]).reshape(X.shape[1], 1) @ (row - mean_vectors[label]).reshape(1, X.shape[1])
            within_class_scatter += scatter_matrix
        overall_mean = np.mean(X, axis=0)
        between_class_scatter = np.zeros((X.shape[1], X.shape[1]))
        for label, data in separated.items():
            n = len(data)
            mean_vec = mean_vectors[label]
            between_class_scatter += n * (mean_vec - overall_mean).reshape(X.shape[1], 1) @ (mean_vec - overall_mean).reshape(1, X.shape[1])
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(within_class_scatter) @ between_class_scatter)
        eigenvectors = eigenvectors.T
        sorted_indexes = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[sorted_indexes]
        if self.n_components is not None:
            self.projection_matrix = sorted_eigenvectors[:self.n_components]
        else:
            self.projection_matrix = sorted_eigenvectors

    """
    The projection of the newly entered data X onto the chosen eigenvectors is returned by the transform() function. (using matrix multiplication).
    """
    def transform(self, X):
        return X @ self.projection_matrix.T
    
    
# Create the LDA object and match the training data to it.
lda = LDA(n_components=2)
lda.fit(X_train, y_train)

# Implement the LDA projection matrix to transform the training and assessment data.
X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)

# Implementing knn with n = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_lda, y_train)
y_pred = knn.predict(X_test_lda)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with LDA pre-processing: {accuracy:.4f}")
