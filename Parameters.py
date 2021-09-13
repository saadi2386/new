# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 22:17:05 2021

@author: Pc Planet
"""
# SVR Hyper Parameter Tuning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston

boston_data = load_boston()

X = boston_data.data
Y = boston_data.target


from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

parameters = {'kernel':['rbf','linear'],'gamma':[1,0.1,0.01]}

grid = GridSearchCV(SVR(),parameters,refit=True,verbose=2,scoring='neg_mean_squared_error')

grid.fit(X,Y)
bset_parameter = grid.best_params_


# K-Mean Hyper Parameter Tuning

k_inertia = []

for i in range(1,10):
    KMNS = KMeans(n_clusters=i,random_state=44)
    KMNS.fit(Data_iris)
    k_inertia.append(KMNS.inertia_)
    

# KNN Hyper Parameter Tuning

from sklearn.datasets import load_iris

iris_Data = load_iris()
X = iris_Data.data
Y = iris_Data.target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=70, test_size=30, 
                                                    random_state=22, shuffle=True,
                                                    stratify=Y)

from sklearn.neighbors import KNeighborsClassifier

KNN_accuracy_train = []
KNN_accuracy_test = []

for k in range(1,50):
    KNN= KNeighborsClassifier(n_neighbors=k, metric='minkowski',p=1)
    KNN.fit(X_train,Y_train)
    KNN_accuracy_test.append(KNN.score(X_test,Y_test))
    KNN_accuracy_train.append(KNN.score(X_train,Y_train))

plt.plot(np.arange(1,50),KNN_accuracy_test,label= 'Test')
plt.plot(np.arange(1,50),KNN_accuracy_train,label= 'Train')
plt.xlabel('Range')
plt.ylabel('KNN_accuracy_test')
plt.legend()
plt.show()








