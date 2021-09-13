# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 22:31:49 2021

@author: Pc Planet
"""

import pandas as pd

Age = pd.Series([15,17,18,20,25,30],index=['Age1','Age2','Age3','Age4','Age5','Age6'])
Age.Age3

filter_age = Age[Age>=18]

Age.values

Age.index

Age.index = ['A1','A2','A3','A4','A5','A6']
Age.index

# data frame

import numpy as np

data = np.array([[18,4,5],[19,6,8],[20,5,7],[17,6,7]]) 

data_frame = pd.DataFrame(data)
data_frame = pd.DataFrame(data,index=['S1','S2','S3','S4'],columns=['Age','Grade1','Grade2'])
data_frame['Grade3'] = [9,8,7,9]

data_frame.loc['S2']

data_frame.iloc[0,3]

data_frame.iloc[:,0]
data_frame.iloc[:,3]
fil_data = data_frame.iloc[:,1:3]

data_frame.drop('Grade1',axis=1)
data_frame.replace(8,0)
data_frame.replace({7:1,9:2})

data_frame.head (2)
data_frame.tail(2)

data_frame.sort_values('Grade1',ascending=True)
data_frame.sort_index(axis=0,ascending=False)

Data = pd.read_csv('Data_Set.csv')













