# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 20:47:11 2021

@author: Pc Planet
"""

import numpy as np
num_array = np.array([[1,2,3],[4,5,6],[7,8,9]])

NPA1 = np.array([[4,6],[10,12]])
NPA2 = np.array([[3,5],[9,11]])

MNP = NPA1@NPA2
MNP1 = np.dot(NPA1, NPA2)

MNP2 = NPA1*NPA2
MNP3 = np.multiply(NPA1, NPA2)
# sum and subract of arrays
sum1 = NPA1+NPA2
sub1 = NPA1-NPA2

sum2 = np.sum(NPA1)
brod_num = NPA1+4

NP2 = np.array([[9,7]])
NPA1+NP2

# divide the array
div = np.divide([12,13,14], 5)
dev = np.floor_divide([12,13,14], 5)

np.math.sqrt(12)

# normal and uniform distribution

ND = np.random.standard_normal((3,4))
UD = np.random.uniform(1,15,(3,4))
# Generate float No. 
ran = np.random.rand(3,5)
# Generate integer No.
Random_array = np.random.randint(1,50,(3,5))

Zero = np.zeros((3,3))
ones = np.ones((3,3))

filter_Ar = np.logical_and(Random_array>30,Random_array<50)
filter_random_Ar = Random_array[filter_Ar]

# main statistics functions

data_np = np.array([1,3,5,7,9,11])
mean_np = np.mean(data_np)
median_np = np.median(data_np)
variance_np = np.var(data_np)
standard_deviation = np.std(data_np)

array_Np = np.array([[1,2,3],[11,12,13]])
variance_array = np.var(array_Np)
variance_array_R = np.var(array_Np,axis=1)
variance_array_C = np.var(array_Np,axis=0)

