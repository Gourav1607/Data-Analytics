#!/usr/bin/env python
# coding: utf-8

# Basics / b-intro.py
# Gourav Siddhad
# 04-Jan-2019

import pandas as pd

df = pd.read_csv('iris.csv',sep=',')
df.drop(['sepal_width'], axis=1)
df.shape
df.head()
df.tail()

print(df.min())
print(df.max())

print(df.min(1))
print(df.max(1))

# try with mean, sum, abs, std
df[0]
df['sepal_length'][0]
df['sepal_length'][0:5]

import numpy as np
np_array = np.array([3,5,25,23])
print(np_array)

set1 = set([3,5,234])
np_array1 = np.array(set1)
print(np_array1)

np_array2 = np.ones((3,4),dtype=int)
print(np_array2)

np_array3 = np.zeros((3,4),dtype=int)
print(np_array3)

np_array4 = np.full((3,4),9)
print(np_array4)

array = [2,45,634,23]
np_array = np.array(array)

array+2
np_array+2

np.add(np_array,2)

a = [[3,4],[5,6]]
b = [[2,5],[7,2]]

a*b

np.matmul(a,b)
np.multiply(a,b)
np.divide(a,b)

np.random.rand()
np.random.rand(4,3)
