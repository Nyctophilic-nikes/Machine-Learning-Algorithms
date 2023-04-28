"""
Implementing fuzzy c means and report the J (objective function) value when c= optimal k.Assume m=2 and beta=0.3.
"""
"""
a function called fuzzy_c_means that performs fuzzy c-means clustering on a dataset X with a specified number of clusters k. 
The other parameters are optional and can be set to default values if not specified
"""
def fuzzy_c_means(X, k, m=2, beta=0.3, max_iter=100):
#initializing the membership matrix randomly, where each row corresponds to a data point and each column corresponds to a cluster.
    U = np.random.rand(X.shape[0], k)
#normalizing the membership matrix so that each row sums to 1.
    U = U / np.sum(U, axis=1, keepdims=True)
#looping that will iterate over each iteration of the fuzzy c-means algorithm, with a maximum number of iterations specified by max_iter.
    for i in range(max_iter):
#computing the centroids from the membership matrix, where each centroid is a weighted average of the data points, weighted by their membership values.
        centroids = np.dot(U.T, X) / np.sum(U, axis=0, keepdims=True).T 
#computing the Euclidean distances between each data point and each centroid
        distances = euclidean_distances(X, centroids)
#updating the membership matrix based on the distances, where each membership value is a function of the distance to the corresponding centroid.
        U_new = np.power(distances, -2/(m-1)) + beta
#normalizing the membership matrix so that each row sums to 1.
        U_new = 1 / np.sum(U_new, axis=1, keepdims=True) * U_new
        U_new = np.maximum(U_new, np.finfo(U_new.dtype).eps)  # avoid division by zero
#checking for convergence by comparing the old and new membership matrices. If they are similar, the algorithm has converged and the loop is exited.
        if np.allclose(U, U_new):
            break
#updating the membership matrix for the next iteration.
        U = U_new
#computing the objective function value, which is a measure of how well the data points are clustered around the centroids
    J = np.sum(np.power(U, m) * distances)
#return the final membership matrix, the final centroids, and the objective function value.
    return U, centroids, J

"""
finding the optimal value of k for fuzzy c-means clustering using the objective function value. 
It calculates the objective function value for different values of k and then plots the results.
"""
# first sets k_values to a range of values from 2 to 10. 
# It then initializes an empty list J_values to store the objective function values for each value of k.
k_values = range(2, 11)
J_values = []
# loops over each value of k and calls the fuzzy_c_means() function to cluster the data using fuzzy c-means with the current value of k. It ignores the membership matrix and centroid values returned by the function and only stores the objective function value in J_values
for k in k_values:
    _, _, J = fuzzy_c_means(X, k)
    J_values.append(J)
plt.plot(k_values, J_values, "bo-")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Objective function value")
plt.show()


# Cluster data using fuzzy c-means with optimal value of k
k = k_values[np.argmin(J_values)]
U, centroids, _ = fuzzy_c_means(X, k)

labels = np.argmax(U, axis=1)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="x")
plt.show()
