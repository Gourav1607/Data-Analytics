#!/usr/bin/env python
# coding: utf-8

# Data Preprocessing / dp-intro.py
# Gourav Siddhad
# 18-Jan-2019

import numpy as np
from sklearn import preprocessing
import pandas as pd

data1 = pd.read_csv("test.csv", header=None)
# link https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

print('Diabetes Dataset: \n')
print(data1.head())

print('Meta Info: \n')
print(data1.describe())

# find the missing values
data1.isnull().sum()

# # Replace with Zero

data2 = data1.fillna(0)
print(data2.head())
print(data2.describe())

# # Replace with Mean

data3 = data1.fillna(data1.mean())
print(data3.head())
print(data3.describe())
data3.isnull().sum()

# # Replace with Median

data4 = data1.fillna(data1.median())
print(data4.head())
print(data4.describe())
data4.isnull().sum()

# # Rescaling the Data

data5 = pd.DataFrame(preprocessing.scale(data4))
data5.describe()

data6 = pd.DataFrame(preprocessing.normalize(data4))
data6.describe()

# Dealing with Categorical Data

enc = preprocessing.OrdinalEncoder()

cdata = pd.DataFrame([['male', 'cricket', 'tea'], [
                     'male', 'football', 'coffee'], ['female', 'chess', 'tea']])
enc.fit(cdata)

enc.transform(cdata)

# # Binarization


cdata2 = np.random.rand(3, 4)*10
print(cdata2)

binarizer = preprocessing.Binarizer(threshold=5)
binarizer.transform(cdata2)

# # Discretization
# go through the sklearn.preprocessing docs for more information.
