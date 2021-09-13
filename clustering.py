# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 19:42:04 2021

@author: Pc Planet
"""

# Clisterin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()
Data_iris = iris.data

# K-Mean clustering

from sklearn.cluster import KMeans

KMNS = KMeans(n_clusters=3)

KMNS.fit(Data_iris)
Labels = KMNS.predict(Data_iris)

center = KMNS.cluster_centers_

plt.scatter(Data_iris[:,2],Data_iris[:,3], c= Labels)
plt.scatter(center[:,2],center[:,3], color='red',s=100)
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal Width (cm)')
plt.grid()
plt.show()

KMNS.inertia_

k_inertia = []

for i in range(1,10):
    KMNS = KMeans(n_clusters=i,random_state=44)
    KMNS.fit(Data_iris)
    k_inertia.append(KMNS.inertia_)
    
plt.plot(range(1,10),k_inertia, color='blue',marker='o')
plt.xlabel('Range')
plt.ylabel('k_inertia')
plt.show()

# DBSCAN

from sklearn.cluster import DBSCAN

DBS = DBSCAN(eps=0.7,min_samples=4)
DBS.fit(Data_iris)
Labels = DBS.labels_

plt.scatter(Data_iris[:,2],Data_iris[:,3], c= Labels)
plt.show()

# Hierarchical Clustering

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

HR = linkage(Data_iris,method='complete')

Dg = dendrogram(HR)

Labels = fcluster(HR,4,criterion='distance')

plt.scatter(Data_iris[:,2],Data_iris[:,3], c=Labels)
plt.show()










