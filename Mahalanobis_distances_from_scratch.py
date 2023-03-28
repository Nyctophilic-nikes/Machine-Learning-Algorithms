import pandas as pd
from scipy.io import arff
import numpy as np

data = arff.loadarff("C:\\Users\\HP\\SML\\Sample data\\ALOI.arff")
#Load data
df = pd.DataFrame(data[0])

df_new = df.drop(columns=["outlier", "id"])

def mahalanobis_scratch(x, mean, inv_cov):
    """
    Computes the Mahalanobis distance between a point x, a mean vector, and an inverse covariance matrix.
    """
    delta = x - mean
    dist = np.sqrt(delta.dot(inv_cov).dot(delta.T))
    return dist
  
# compute the mean and covariance matrix
mean = np.mean(df_new, axis=0)
cov = np.cov(df_new.T)
inv_cov = np.linalg.inv(cov)

# compute the Mahalanobis distance for each row
distances = []
for i in range(len(df_new)):
    row = df_new.iloc[i].values
    distances.append(mahalanobis_scratch(row, mean, inv_cov))

# add the distances to the DataFrame
df_new["mahalanobis_distance"] = distances

display(df_new)
