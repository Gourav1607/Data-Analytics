#!/usr/bin/env python
# coding: utf-8

# Statistics / statistics.py
# Gourav Siddhad
# 11-Jan-2019

# 1. Load the 'test.arff' file and select the first 10 features for further processing.
# 2. Find the co-variance matrix of the dataset.
# 3. Plot the scatter plot for each feature(attribute) combination. [ plot the graphs in subplots]
# 4. Calculate the correlation matrix of the dataset.
# 5. Comment on the relationship between scatter plots and correlation values.

# [install 'liac-arff' to work with .arff file]
# Load the 'test.arff' file and select the first 10 features for further processing.

import matplotlib.pyplot as plt
from scipy.io import arff
import numpy as np
import pandas as pd

data = arff.loadarff('test.arff')
df = pd.DataFrame(data[0][['ACAIC', 'ACMIC', 'AID', 'ANA',
                           'CAM', 'CBO', 'CBOin', 'CBOout', 'CIS', 'CLD']])
matrix = np.matrix(df)
mat = matrix.astype(int)
print(mat)

# Find the co-variance matrix of the dataset.


def findCovariance(x, y):
    meanx = np.mean(x)
    meany = np.mean(y)

    diffx = x-meanx
    diffy = y-meany

    return np.sum(np.dot(diffx, diffy))/(len(x)-1)


result = []
for p in range(0, 10):
    x = mat[:, p]
    temp = []
    for d in range(0, 10):
        y = mat[:, d].T
        temp.append(findCovariance(x, y))
    result.append(temp)

for i in range(0, 10):
    print(result[i])

# Plot the scatter plot for each feature(attribute) combination. [ plot the graphs in subplots]


f, ax = plt.subplots(10, 10)
f.set_size_inches(18.5, 10.5)
for i in range(0, 10):
    x = np.array(mat[:, i])
    for j in range(0, 10):
        y = np.array(mat[:, j])
        ax[i, j].scatter(x, y)

f.subplots_adjust(top=0.92, bottom=0.08, left=0.10,
                  right=0.95, hspace=0.25, wspace=0.35)
plt.show()

# Calculate the correlation matrix of the dataset.


def findMean(data):
    return np.sum(data)/len(data)


def findVariance(data):
    mean = findMean(data)
    diff = data-mean
    return np.sum(diff**2)/(len(data)-1)


def findSD(data):
    return np.sqrt(findVariance(data))


def findCorrelation(x, y):
    return findCovariance(x, y)/(findSD(x)*findSD(y))


tempr = []
res = []
for i in range(0, 10):
    x = np.array(mat[:, i])
    for j in range(0, 10):
        y = np.array(mat[:, j].T)
        try:
            tempr.append(findCorrelation(x, y))
        except:
            print('Error')
    res.append(tempr)
    tempr = []

for i in range(0, 10):
    print(res[i])
