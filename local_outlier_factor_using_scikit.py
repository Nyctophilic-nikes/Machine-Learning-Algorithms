import pandas as pd
from scipy.io import arff
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

data = arff.loadarff("C:\\Users\\HP\\SML\\Sample data\\ALOI.arff")

#Load data
df = pd.DataFrame(data[0])

df_new = df.drop(columns=["outlier", "id"])

# initialize the LOF model
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

# fit the model and make predictions
y_pred = lof.fit_predict(df_new)

# add the predictions to the DataFrame
df_new["lof"] = y_pred

# display the DataFrame
display(df_new)
