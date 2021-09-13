# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 19:18:20 2021

@author: Pc Planet
"""
# Regression

from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as pd

boston_data = load_boston()

X = boston_data.data
Y = boston_data.target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size= 0.8,test_size=0.2,
                                                    random_state=80)
from sklearn.preprocessing import MinMaxScaler

scale = MinMaxScaler(feature_range=(0,1))
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)

Y_train = Y_train.reshape(-1,1)
Y_train = scale.fit_transform(Y_train)

# MLR (Multiple Linear Regression)

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()
linear_reg.fit(X_train,Y_train)
Prediction_reg = linear_reg.predict(X_test)

Prediction_reg = scale.inverse_transform(Prediction_reg)


# Evaluation Metrics

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from  sklearn.metrics import r2_score

import math

mean_absolute_error(Y_test ,Prediction_reg)
mean_squared_error(Y_test , Prediction_reg)
R2 = r2_score(Y_test , Prediction_reg)

math.sqrt(mean_squared_error(Y_test , Prediction_reg))


def mean_absolute_percentage_error (y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean (np.abs(y_true-y_pred)/y_true)*100

mean_absolute_percentage_error(Y_test,Prediction_reg)


'''
def mean_absolute_error (y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean (np.abs(y_pred-y_true)/102)

mean_absolute_error(Y_test,Prediction_reg)

def mean_squared_error (y_true, y_pred):
      y_true = np.array(y_true)
      y_pred = np.array(y_pred)
      return np.mean (np.abs(y_pred-y_true)**2)/102
mean_squared_error(Y_test,Prediction_reg)
'''

# PLR (Polynomial Linear Regression)

from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as pd

boston_data = load_boston()

X = boston_data.data[:,5]
Y = boston_data.target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size= 0.8,test_size=0.2,
                                                    random_state=80)

from sklearn.preprocessing import PolynomialFeatures

Poly_F = PolynomialFeatures(degree=2)
X_train = X_train.reshape(-1,1)
Poly_X = Poly_F.fit_transform(X_train)

from sklearn.linear_model import LinearRegression

Linear_reg = LinearRegression()
Poly_Linear_reg = Linear_reg.fit(Poly_X,Y_train)
X_test = X_test.reshape(-1,1)
Poly_XT = Poly_F.fit_transform(X_test)
Predict_PLR = Poly_Linear_reg.predict(Poly_XT)

from sklearn.metrics import r2_score

R2 = r2_score(Y_test,Predict_PLR)

# Random Forest

from sklearn.ensemble import RandomForestRegressor

Random_F = RandomForestRegressor(random_state=33)
Random_F.fit(X_train,Y_train)

Predicted_val_RF = Random_F.predict(X_test)

Predicted_val_RF = Predicted_val_RF.reshape(-1,1)
Predicted_val_RF = scale.inverse_transform(Predicted_val_RF)


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from  sklearn.metrics import r2_score

import math

mean_absolute_error(Y_test ,Predicted_val_RF)
mean_squared_error(Y_test , Predicted_val_RF)
R2 = r2_score(Y_test , Predicted_val_RF)

math.sqrt(mean_squared_error(Y_test , Predicted_val_RF))


def mean_absolute_percentage_error (y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean (np.abs(y_true-y_pred)/y_true)*100

mean_absolute_percentage_error(Y_test,Predicted_val_RF)


# SVR

from sklearn.svm import SVR

Regressor_SVR = SVR(kernel="rbf") 
Regressor_SVR.fit(X_train,Y_train)

Predict_SVR = Regressor_SVR.predict(X_train)
Predict_SVR = Predict_SVR.reshape(-1,1)
Predict_SVR = scale.inverse_transform(Predict_SVR)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
import math

mean_absolute_error(Y_test, Predict_SVR)
math.sqrt(mean_squared_error(Y_test, Predict_SVR))
mean_absolute_percentage_error(Y_test, Predict_SVR)
R2 = r2_score(Y_test, Predict_SVR)












