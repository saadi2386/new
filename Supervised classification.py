# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 03:00:46 2021

@author: Pc Planet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()
iris.feature_names
iris.target_names

iris_Data= iris.data
iris_Data= pd.DataFrame(iris_Data,columns= iris.feature_names)
iris_Data['labels']= iris.target

plt.scatter(iris_Data.iloc[:,0],iris_Data.iloc[:,1], c = iris.target)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.grid()
plt.show()

x = iris_Data.iloc[:,0:4]
y = iris_Data.iloc[:,4]

# K-NN Classifiar

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors= 6,metric = 'minkowski',p= 1)
KNN.fit(x,y)

x_N1 = np.array([[5,3.2,1,0.3]])
KNN.predict(x_N1)

x_N2 = np.array([[4.0,4,2,1]])
KNN.predict(x_N2)

x_N3 = np.array([[6,4.5,3.8,2.9]])
KNN.predict(x_N3)

x_N4 = np.array([[9,8.5,7.9,7.1]])
KNN.predict(x_N4)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x,y, train_size=0.8,test_size=0.2,
                                                    random_state=90,shuffle= True,
                                                    stratify=y) 

KNN = KNeighborsClassifier(n_neighbors=10, metric= 'minkowski',p=1)
KNN.fit(X_train,Y_train)
prediction = KNN.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,prediction)

# Decision tree

from sklearn.tree import DecisionTreeClassifier

Dt = DecisionTreeClassifier()
Dt.fit(X_train,Y_train)
prediction_Dt = Dt.predict(X_test)
accuracy_score(Y_test,prediction_Dt)

from sklearn.model_selection import cross_val_score
Score_Dt = cross_val_score(Dt, x, y, cv = 10)

# Naive Bayes Classification

from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
NB.fit(X_train,Y_train)
Prediction_NB = NB.predict(X_test)

accuracy_score(Y_test,Prediction_NB)

Score_NB = cross_val_score(NB, x, y, cv = 10)


# Logistic regression 
import pandas as pd

from sklearn.datasets import load_breast_cancer


Cancer_Data = load_breast_cancer()

X_target = Cancer_Data.data
Y_target = Cancer_Data.target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_target, Y_target,train_size=0.7,
                                                    test_size=0.3,random_state=85)

from sklearn.linear_model import LogisticRegression

Lr = LogisticRegression()
Lr.fit(X_train,Y_train)

Prediction_Lr = Lr.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(Y_test,Prediction_Lr)

from sklearn.model_selection import cross_val_score

Score_Lr = cross_val_score(Lr,X_target,Y_target, cv=10)

# Evaluation Metrices

from sklearn.metrics import confusion_matrix,classification_report

conf_Matrix = confusion_matrix(Y_test,Prediction_Lr)
class_Report = classification_report(Y_test,Prediction_Lr)


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

Y_prob = Lr.predict_proba(X_test)
Y_prob = Y_prob[:,1]

FPR , TPR , Thresholds = roc_curve(Y_test,Y_prob)
 
plt.plot (FPR,TPR)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

from sklearn.metrics import roc_auc_score

roc_auc_score(Y_test,Y_prob)



