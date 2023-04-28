"""
Implement k-means clustering and silhouette analysis algorithms to cluster
the given data and find the optimal ‘k’ value.
"""

# importing modules
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

"""
k-means clustering function and imports the silhouette score function from scikit-learn.
function called k_means that takes in three arguments: X, a 2D array of shape (n_samples, n_features) containing the data to be clustered; k, an integer representing the number of clusters to form; and max_iter, an integer representing the maximum number of iterations to run the k-means algorithm. The default value of max_iter is set to 100.
initializes k centroids randomly by selecting k samples from X without replacement using the np.random.choice function.
a loop that will run for max_iter iterations at most.
calculates the Euclidean distance between each point in X and each centroid using the euclidean_distances function from scikit-learn.
assigns each point to the closest centroid by finding the index of the minimum distance for each point along the axis=1 (i.e., across the columns).
initializes a new array to hold the updated centroids.
a loop that will iterate over each cluster
updates the centroid for the current cluster by taking the mean of all the points assigned to that cluster.
checks if the centroids have converged by comparing the new centroids to the previous centroids using the np.allclose function.
checks if the centroids have converged by comparing the new centroids to the previous centroids using the np.allclose function.
updates the centroids to the new centroids.
returns the final cluster assignments and centroids as a tuple. imports the silhouette score function from scikit-learn, which can be used to evaluate the quality of the clustering results.
"""
def k_means(X, k, max_iter=100):
# initializing k centroids randomly by selecting k samples from X without replacement using the np.random.choice function.
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
# looping that will run for max_iter iterations at most.    
    for i in range(max_iter):
        # Assign each point to the closest centroid
        distances = euclidean_distances(X, centroids)
#assigning each point to the closest centroid by finding the index of the minimum distance for each point along the axis=1 (i.e., across the columns).
        labels = np.argmin(distances, axis=1)
#updating the centroid for the current cluster by taking the mean of all the points assigned to that cluster.
        new_centroids = np.empty_like(centroids)
#looping that will iterate over each cluster
        for j in range(k):
            new_centroids[j] = np.mean(X[labels == j], axis=0)
# checking if the centroids have converged by comparing the new centroids to the previous centroids using the np.allclose function.
        if np.allclose(centroids, new_centroids):
            break
#updating the centroids to the new centroids
        centroids = new_centroids
# return the final cluster assignments and centroids as a tuple. imports the silhouette score function from scikit-learn, which can be used to evaluate the quality of the clustering results.
    return labels, centroids


"""
a function called silhouette that takes in two arguments: X, a 2D array of shape (n_samples, n_features) containing the data to be clustered; 
and k_values, a list of integers representing the number of clusters to form. 
The function returns a list of silhouette scores for each value of k
"""
from sklearn.metrics import silhouette_score
def silhouette(X, k_values):
#initializing an empty list to hold the silhouette scores.
    scores = []
    
    for k in k_values:
# calling the k_means function to cluster the data into k clusters and assigns each point to the closest centroid. 
# The function return the final cluster assignments and centroids, but we only need the cluster assignments 
# so we use the underscore to ignore the centroids.
        labels, _ = k_means(X, k)
# calculating the silhouette score for the current clustering solution using the silhouette_score function from scikit-learn
        score = silhouette_score(X, labels)
# adding the current silhouette score to the list of scores.
        scores.append(score)
# return the list of silhouette scores.
    return scores

import matplotlib.pyplot as plt

"""
performing silhouette analysis to find the optimal number of clusters (k) for K-means clustering. 
It loads the data from a .npy file, normalizes it, and then calls the silhouette() function with a range of k values from 2 to 10. 
The resulting silhouette scores for each k value are plotted as a function of k.
"""
#Loading data
X = np.load("C:\\Users\\HP\\Downloads\\kmeans_data.npy")
# Normalizing the data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# Finding optimal value of k using silhouette analysis
k_values = range(2, 11)
scores = silhouette(X, k_values)
# plotting the data
plt.plot(k_values, scores, "bo-")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.show()

"""
clustering the data using K-means with the optimal value of k that was found using silhouette analysis. 
It assigns each data point to its nearest centroid and then plots the resulting clusters
"""
# first sets k to the optimal value of k that was found using the silhouette scores. 
# It then calls the k_means() function to cluster the data, passing in X and k. 
# This function returns the cluster labels and the centroids.
k = k_values[np.argmax(scores)]
labels, centroids = k_means(X, k)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x")
plt.show()
