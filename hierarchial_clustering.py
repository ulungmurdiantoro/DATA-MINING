import matplotlib_inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib_inline

from sklearn import datasets
iris = datasets.load_iris()

df=pd.DataFrame(iris['data'])
print(df.head())

from scipy.cluster.vq import whiten
scaled_data = whiten(df.to_numpy())

from scipy.cluster.hierarchy import fcluster, linkage
distance_matrix = linkage(scaled_data, method='ward', metric='euclidean')

from scipy.cluster.hierarchy import dendrogram
dn =dendrogram(distance_matrix)
plt.show()