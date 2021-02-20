#!/usr/bin/env python
# coding: utf-8

# Data Preprocessing / datapreprocessing.py
# Gourav Siddhad
# 18-Jan-2019

# e1. Load the iris.csv file and perform following steps to fill missing values.
import matplotlib.pyplot as plt
# 1. replace with highest freq data of that column
# 2. change to categorical
# 3. replace with classwise mean, median

import numpy as np
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("iris.csv", header=None)

print('Iris Dataset: \n')
data1 = data.drop(0)
print(data1.head())
print()
data1.describe()

# 1. replace with highest freq data of that column

data2 = data1
for i in range(0, 5):
    p = data1[i].value_counts().idxmax()
    data2[i] = data1[i].fillna(p)

print(data2.head())
data2.describe()

# 2. change to categorical
# Discretize it according to range

'''
bins = [0, 1, 2, 2.5, 4.4, 6.9, 8, 100]
labels = [0.5, 1.5, 2.25, 3.45, 5.55, 7.45, 10]
'''

data3 = np.asmatrix(data1, dtype='float')
p = 0
for i in range(0, 150):
    for j in range(0, 5):
        q = data3[i, j]
        if q > 0 and q < 1:
            p = 0.5
        elif q > 1 and q < 2:
            p = 1.5
        elif q > 2 and q < 2.5:
            p = 2.25
        elif q > 2.5 and q < 4.4:
            p = 3.45
        elif q > 4.4 and q < 6.9:
            p = 5.55
        elif q > 6.9 and q < 8:
            p = 7.45
        else:
            p = 10
        data3[i, j] = p

data3 = pd.DataFrame(data3)
print(data3.head())

# 3. replace with classwise mean, median

data4 = np.asmatrix(data1, dtype='float')
for j in range(0, 4):
    m0, m1, m2 = 0, 0, 0
    c, t0, t1, t2 = 0, 0, 0, 0
    for i in range(0, 150):
        c = data4[i, 4]
        q = data4[i, j]
        if not np.isnan(q):
            if c is 0:
                m0 += q
                t0 += 1
            elif c is 1:
                m1 += q
                t1 += 1
            else:
                m2 += q
                t2 += 1

    for i in range(0, 150):
        if t0 > 0:
            t0 = m0/t0
        if t1 > 0:
            t1 = m1/t1
        if t2 > 0:
            t2 = m2/t2
        c = data4[i, 4]
        q = data4[i, j]
        if np.isnan(q):
            if c is 0:
                data4[i, j] = t0
            elif c is 1:
                data4[i, j] = t1
            else:
                data4[i, j] = t2
data4 = pd.DataFrame(data4)
print('Mean')
print(data4.head())
print('\nMedian')

data5 = np.asmatrix(data1, dtype='float')
for j in range(0, 4):
    m0, m1, m2 = [], [], []
    c = 0
    for i in range(0, 150):
        c = data5[i, 4]
        q = data5[i, j]
        if not np.isnan(q):
            if c is 0:
                m0.append(q)
            elif c is 1:
                m1.append(q)
            else:
                m2.append(q)

    for i in range(0, 150):
        t0 = np.array(m0)
        t1 = np.array(m1)
        t2 = np.array(m2)
        c = data5[i, 4]
        q = data5[i, j]
        if np.isnan(q):
            if c is 0:
                data5[i, j] = np.median(t0)
            elif c is 1:
                data5[i, j] = np.median(t1)
            else:
                data5[i, j] = np.median(t2)

data5 = pd.DataFrame(data5)
print(data5.head())

# e2. Plot (using suitable plot) the mean, std, min, max for each preprocessing approach
# to compare the effect of preprocessing approach.
# [use the example file ]


# Read Data
data = pd.read_csv("test.csv", header=None)

# Replace with Zero
data1 = data.fillna(0)

# Replace with Mean
data2 = data.fillna(data.mean())

# Replace with Median
data3 = data.fillna(data.median())

# Rescaling the Data
data4 = pd.DataFrame(preprocessing.scale(data3))

# Normalize the Data
data5 = pd.DataFrame(preprocessing.normalize(data3))


f, ax = plt.subplots(2, 2)
f.set_size_inches(18.5, 10.5)
dorg, d1, d2, d3, d4, d5 = [], [], [], [], [], []
for i in range(0, 9):
    dorg.append(np.mean(data[i].astype('float')))
    d1.append(np.mean(data1[i].astype('float')))
    d2.append(np.mean(data2[i].astype('float')))
    d3.append(np.mean(data3[i].astype('float')))
    d4.append(np.mean(data4[i].astype('float')))
    d5.append(np.mean(data5[i].astype('float')))

ax[0, 0].plot(dorg, label='Original')
ax[0, 0].plot(d1, label='Zero')
ax[0, 0].plot(d2, label='Mean')
ax[0, 0].plot(d3, label='Median')
ax[0, 0].plot(d4, label='Re-Scaled')
ax[0, 0].plot(d5, label='Normalized')
ax[0, 0].legend()

dorg, d1, d2, d3, d4, d5 = [], [], [], [], [], []
for i in range(0, 9):
    dorg.append(np.std(data[i].astype('float')))
    d1.append(np.std(data1[i].astype('float')))
    d2.append(np.std(data2[i].astype('float')))
    d3.append(np.std(data3[i].astype('float')))
    d4.append(np.std(data4[i].astype('float')))
    d5.append(np.std(data5[i].astype('float')))

ax[0, 1].plot(dorg, label='Original')
ax[0, 1].plot(d1, label='Zero')
ax[0, 1].plot(d2, label='Mean')
ax[0, 1].plot(d3, label='Median')
ax[0, 1].plot(d4, label='Re-Scaled')
ax[0, 1].plot(d5, label='Normalized')
ax[0, 1].legend()

dorg, d1, d2, d3, d4, d5 = [], [], [], [], [], []
for i in range(0, 9):
    dorg.append(np.min(data[i].astype('float')))
    d1.append(np.min(data1[i].astype('float')))
    d2.append(np.min(data2[i].astype('float')))
    d3.append(np.min(data3[i].astype('float')))
    d4.append(np.min(data4[i].astype('float')))
    d5.append(np.min(data5[i].astype('float')))

ax[1, 0].plot(dorg, label='Original')
ax[1, 0].plot(d1, label='Zero')
ax[1, 0].plot(d2, label='Mean')
ax[1, 0].plot(d3, label='Median')
ax[1, 0].plot(d4, label='Re-Scaled')
ax[1, 0].plot(d5, label='Normalized')
ax[1, 0].legend()

dorg, d1, d2, d3, d4, d5 = [], [], [], [], [], []
for i in range(0, 9):
    dorg.append(np.max(data[i].astype('float')))
    d1.append(np.max(data1[i].astype('float')))
    d2.append(np.max(data2[i].astype('float')))
    d3.append(np.max(data3[i].astype('float')))
    d4.append(np.max(data4[i].astype('float')))
    d5.append(np.max(data5[i].astype('float')))

ax[1, 1].plot(dorg, label='Original')
ax[1, 1].plot(d1, label='Zero')
ax[1, 1].plot(d2, label='Mean')
ax[1, 1].plot(d3, label='Median')
ax[1, 1].plot(d4, label='Re-Scaled')
ax[1, 1].plot(d5, label='Normalized')
ax[1, 1].legend()

ax[0, 0].set_xlabel('Feature')
ax[0, 0].set_ylabel('Values')
ax[0, 0].set_title('Mean')

ax[0, 1].set_xlabel('Feature')
ax[0, 1].set_ylabel('Values')
ax[0, 1].set_title('Standard Deviation')

ax[1, 0].set_xlabel('Feature')
ax[1, 0].set_ylabel('Values')
ax[1, 0].set_title('Minimum')

ax[1, 1].set_xlabel('Feature')
ax[1, 1].set_ylabel('Values')
ax[1, 1].set_title('Maximum')

f.subplots_adjust(top=0.92, bottom=0.08, left=0.10,
                  right=0.95, hspace=0.25, wspace=0.35)
plt.show()
