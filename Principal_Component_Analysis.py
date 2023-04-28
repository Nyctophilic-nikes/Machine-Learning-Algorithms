"""
Steps to follow:
a) Download the dataset from here: https://www.kaggle.com/datasets/scolianni/mnistasjpg
b) Load the image dataset in your environment and convert it into a suitable format for creating an ML model.
c) Implement PCA from scratch.
d) Use kNN to train ML models on the training set of both original featuresand the transformed features (number of PCs = 5, 25, 125) obtained from
PCA. Run the trained models on the test set and report the classification accuracies.
e) Plot explained-variance (ratio of eigenvalue and sum of all eigenvalues) v/s PCs.
"""

# Importing packages
from imutils import paths
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

# loading training dataset
training_Set = "C:\\Users\\HP\\Downloads\\archive\\trainingSet\\trainingSet"
imagePaths = list(paths.list_images(training_Set))

"""
loading and preprocessing the image data.
The images are read using cv2.imread, converted to grayscale using cv2.cvtColor, and then resized to a fixed size of 64 x 64 using cv2.resize. 
The resulting grayscale image is flattened into a 1D array using flatten, and the flattened image data is stored in the array X. 
The corresponding labels for the images are stored in the array y.
"""
X = []
y = []

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    X.append(resized.flatten())
    y.append(imagePath.split(os.path.sep)[-2])
    
X = np.array(X)
y = np.array(y)

"""
The __init__ method initializes the PCA object with a number of components n_components. 
The fit method takes a data matrix X and computes the principal components and eigenvalues of the covariance matrix. 
The mean of the data is subtracted from the data matrix to center it around the origin. 
The covariance matrix is computed using np.cov, and the eigenvalues and eigenvectors are obtained using np.linalg.eigh. 
The eigenvalues and eigenvectors are sorted in descending order of the eigenvalues, and the top n_components eigenvectors are stored as the principal components. 
The transform method takes a data matrix X and projects it onto the principal components to obtain a lower-dimensional representation of the data. 
The centered data matrix is multiplied by the principal components using np.dot to obtain the transformed data.
"""
class PCA:
# The __init__ method initializes the PCA object with a number of components n_components. 
    def __init__(self, n_components):
        self.n_components = n_components
# The fit method takes a data matrix X and computes the principal components and eigenvalues of the covariance matrix.
    def fit(self, X):
# The mean of the data is subtracted from the data matrix to center it around the origin.
        self.mean = np.mean(X.astype(float), axis=0)
        X_centered = X.astype(float) - self.mean
# The covariance matrix is computed using np.cov, and the eigenvalues and eigenvectors are obtained using np.linalg.eigh.
        cov_matrix = np.cov(X_centered, rowvar=False)
# The eigenvalues and eigenvectors are sorted in descending order of the eigenvalues, and the top n_components eigenvectors are stored as the principal components. 
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[indices][:self.n_components]
        self.eigenvectors = eigenvectors[:, indices][:, :self.n_components]
# The transform method takes a data matrix X and projects it onto the principal components to obtain a lower-dimensional representation of the data. 
    def transform(self, X):
# The centered data matrix is multiplied by the principal components using np.dot to obtain the transformed data.
        X_centered = X.astype(float) - self.mean
        return np.dot(X_centered, self.eigenvectors)
  
  """
splitting the data into training and testing sets, with a test size of 20% and a random state of 42. 
It assigns the training data to X_train and y_train, and the testing data to X_test and y_test.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
performing Principal Component Analysis (PCA) on the training and testing data for different numbers of principal components (5, 25, and 125).
For each value of n_components, the code initializes a PCA object with n_components as the parameter, fits it on the training data using the fit method, and transforms both the training and testing data using the transform method.
After the data is transformed, the code creates a KNeighborsClassifier object with 5 neighbors, fits it on the transformed training data using the fit method, predicts the labels of the transformed testing data using the predict method, and calculates the accuracy of the predicted labels using the accuracy_score function. The accuracy is then printed to the console
"""
for n_components in [5, 25, 125]:
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_transformed = pca.transform(X_train)
    X_test_transformed = pca.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_transformed, y_train)
    y_pred = knn.predict(X_test_transformed)
    acc = accuracy_score(y_test, y_pred)
    print(f"kNN with {n_components} PCA features: {acc:.4f}")
    
   
  """
perform PCA on the training data and get information on the amount of variance explained by each principal component, as well as the cumulative amount of variance explained by all components.
The n_components parameter is set to the number of features in the training data, which means that the PCA will try to find as many principal components as there are features.
fits the PCA model to the training data X_train. During this process, the model calculates the principal components and their corresponding eigenvalues.
calculates the normalized eigenvalues, which represent the proportion of variance explained by each principal component. The eigenvalues attribute of the PCA object contains the raw eigenvalues. 
Dividing these by the sum of all eigenvalues gives you the proportion of variance explained by each component.
calculates the cumulative proportion of variance explained by the principal components. 
The cumsum function from numpy is used to calculate the cumulative sum of the eigenvalues array, which gives you an array of the same length that represents the cumulative proportion of variance explained by each additional principal component
"""
pca = PCA(n_components=X_train.shape[1])
pca.fit(X_train)
eigenvalues = pca.eigenvalues / np.sum(pca.eigenvalues)
cumulative_var = np.cumsum(eigenvalues)

"""
plot to visualize the proportion of variance explained by each principal component and the cumulative proportion of variance explained by all components, up to the number of components considered.
"""
plt.plot(range(1, len(eigenvalues)+1), eigenvalues, label='Explained variance')
plt.plot(range(1, len(eigenvalues)+1), cumulative_var, label='Cumulative explained variance')
plt.axhline(y=0.8, color='r', linestyle='--', label='80% variance threshold')
plt.xlabel('Number of principal components')
plt.ylabel('Explained variance ratio')
plt.legend()
plt.show()
