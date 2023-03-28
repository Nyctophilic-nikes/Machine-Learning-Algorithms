import pandas as pd
from scipy.io import arff
from scipy.spatial.distance import mahalanobis
import numpy as np

#Loading the data from sample folder
data = arff.loadarff("C:\\Users\\HP\\SML\\Sample data\\ALOI.arff")
df = pd.DataFrame(data[0])

#Dropping the columns
df_new = df.drop(columns=["outlier", "id"])

# computing the mean and covariance matrix
mean = np.mean(df_new, axis=0)
cov = np.cov(df_new.T)

# computing the Mahalanobis distance for each row
distances = []
for i in range(len(df_new)):
    row = df_new.iloc[i].values
    distances.append(mahalanobis(row, mean, np.linalg.inv(cov)))
    
#  Adding new column in the data
df_new["mahalanobis_distance"] = distances

display(df_new)
