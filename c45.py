from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
features = iris['data']
target = iris['target']

decisiontree = DecisionTreeClassifier(random_state=0,
                                      max_depth=None, min_samples_split=2,
                                      min_samples_leaf=1, min_weight_fraction_leaf=0,
                                      max_leaf_nodes=None, min_impurity_decrease=0)
model = decisiontree.fit(features, target)
observation = [[5,4,3,2]]
model.predict(observation)
model.predict_proba(observation)

import pydotplus
from sklearn import tree
dot_data = tree.export_graphviz(decisiontree, out_file=None,
                                feature_names=iris['feature_names'], class_names=iris['target_names'])
from IPython.display import Image
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png("E:\iris.png")

import numpy as np
import pandas as pd
from sklearn import tree

irisDataset = pd.read_csv('klasifikasi_dataset_iris.csv', delimiter=',', header=0)
irisDataset["Species"] = pd.factorize(irisDataset.species)[0]
irisDataset = irisDataset.drop(labels="Id", axis=1)

irisDataset = irisDataset.to_numpy()

dataTraining = np.concatenate((irisDataset[0:40,:], irisDataset[50:90,:]), axis=0)
dataTesting = np.concatenate((irisDataset[40:50,:], irisDataset[90:100,:]), axis=0)

inputTraining = dataTraining[:,0:4]
inputTesting = dataTesting[:,0:4]
labelTraining = dataTraining[:,4]
labelTesting = dataTesting[:,4]

model = tree.DecisionTreeClassifier()
model = model.fit(inputTraining, labelTraining)
hasilPrediksi = model.predict(inputTesting)
print("label sebenarnya", labelTesting)
print("hasil prediksi:  ", hasilPrediksi)

prediksiBenar = (hasilPrediksi == labelTraining).sum()
prediksiSalah = (hasilPrediksi != labelTesting).sum()
print("prediksi benar: ", prediksiBenar, " data")
print("prediksi salah: ", prediksiSalah, " data")
print("akurasi: ", prediksiBenar/(prediksiBenar+prediksiSalah)
      * 100, "%")

