# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 00:16:53 2021

@author: Pc Planet
"""

import pandas as pd

Data_set1= pd.read_csv("New_Data.csv")

Data_set2= pd.read_csv("New_Data.csv",header=2)
Data_set3= Data_set2.rename(columns={"Temperature":"Temp"})
Data_set3.drop('No. Occupants',axis= 1, inplace = True)

Data_set4 = Data_set3.drop(2,axis=0)
Data_set5 = Data_set4.reset_index(drop= True)
Data_set5.describe()
Min_item = Data_set5["E_Heat"].min()
Data_set5['E_Heat'][Data_set5['E_Heat']==Min_item]
Data_set5['E_Heat'].replace(-4,21, inplace = True)

# covariance
Data_set5.cov()

import seaborn as sns
sns.heatmap(Data_set5.corr())

# Missing value
Data_set5.info()

import numpy as np

Data_set6= Data_set5.replace('!', np.NaN)
Data_set6.info()
Data_set6 = Data_set6.apply(pd.to_numeric)
Data_set6.info()

Data_set6.isnull()
Data_set6.drop(13,axis=0, inplace = True)
Data_set6.dropna(axis=0, inplace = True)
Data_set7 = Data_set6.fillna(method = 'ffill')

from sklearn.impute import SimpleImputer

M_var = SimpleImputer(missing_values = np.nan, strategy = 'mean')
M_var.fit(Data_set6)
Data_set8 = M_var.transform(Data_set6)

# Outlier detection

Data_set7.boxplot()

Data_set7['E_Plug'].quantile(0.25)
Data_set7['E_Plug'].quantile(0.75)

'''
Q1 = 21.25
Q3 = 33.75
IQR = Q3 - Q1 = 12.5

Mild Outlier
Lower Bound = Q1-1.5*IQR = 2.5
Upper Bound = Q3+1.5*IQR = 52.5

Extreme Outlier
lower Outlier = Q1-3*IQR = -16.25
upper Outlier = Q3+3*IQR = 71.25

'''

Data_set7["E_Plug"].replace(120,50,inplace = True)


# Concatination

New_col = pd.read_csv("Data_New.csv")

Data_set9 = pd.concat([Data_set7,New_col],axis=1)

# Dummy Variables

Data_set9.info()
Data_set10 = pd.get_dummies(Data_set9)
Data_set10.info()


# Normalization

from sklearn.preprocessing import minmax_scale, normalize

# First min-max sale
Data_set11 = minmax_scale(Data_set10, feature_range=(0,1))

Data_set12 = normalize(Data_set10, norm= "l2",axis=0)
# axis=0 is for normalizing features/ axis=1 is for normalizing each sample

Data_set13 = pd.DataFrame(Data_set12,columns=['Time','E_Plug','E_Heat',
                                              'Price','Temp','OffPeak','Peak'])







