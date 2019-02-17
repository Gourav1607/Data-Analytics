#!/usr/bin/env python
# coding: utf-8

# Basics / basics.py
# Gourav Siddhad
# 04-Jan-2019

# Find the following statistics for each column (except first column)
# min, max, mean

import pandas as pd
df = pd.read_csv('wine.csv',sep=',')
print('Min')
print(df.min())
print('\n\nMax')
print(df.max())
print('\n\nMean')
print(df.mean())

# Write a Numpy based menu driven program with following options:
# option 1. Enter list data
# option 2. Find Maximum in the data
# option 3. sort the data
# option 4. find the Median of the data

import numpy as np
list = []
ch = 1

def getdata():
    n = int(input('Enter Number of Data : '))
    print('Enter Data : ')
    for i in range(0,n):
        list.append(int(input('%d : '%(i+1))))

while ch != 0:
    print('Menu')
    print('1. Enter list data')
    print('2. Find Maximum in the data')
    print('3. Sort the data')
    print('4. Find the Median of the data')
    print('0. Exit')
    ch = int(input('Enter Choice '))

    if ch == 1:
        getdata()
    elif ch == 2:
        print(np.max(list))
    elif ch == 3:
        print(np.sort(list))
    elif ch == 4:
        print(np.median(list))

# Year wise GDP data of three countries given below
# years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
# country1_gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]
# country2_gdp = [30.2, 240.3, 675.9, 1262.5, 3579.6, 7089.7, 10958.3]
# country3_gdp = [10.2, 27.3, 65.9, 362.5, 359.6, 789.7, 1058.3]
# Plot the line graph which shows the year wise performance of each country.

from matplotlib import pyplot as plt
fig = plt.figure()

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
country1_gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]
country2_gdp = [30.2, 240.3, 675.9, 1262.5, 3579.6, 7089.7, 10958.3]
country3_gdp = [10.2, 27.3, 65.9, 362.5, 359.6, 789.7, 1058.3]

plt.plot(years, country1_gdp, marker='*')
plt.plot(years, country2_gdp, marker='o')
plt.plot(years, country3_gdp, marker='x')
plt.show()

# You have iris.csv file. Perform following tasks.
# 1.1 Load the iris dataset.
# 2.2 Make pair of two-two features and plot the 2-D scatter plots.
# 2.3 Make pair of three-three features and plot 3-D scatter plots.
# 2.4 Try to infer information from the scatter plots. (what could be best set of features for classification of data)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('iris.csv',sep=',')

a = df['sepal_length']
b = df['sepal_width']
c = df['petal_length']
d = df['petal_width']

N=150

colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

f1, axarr = plt.subplots(2, 3)
axarr[0, 0].scatter(a, b, c=colors, alpha=0.5)
axarr[0, 0].set_title('SL     SW')
axarr[0, 1].scatter(a, c, c=colors, alpha=0.5)
axarr[0, 1].set_title('SL     PL')
axarr[0, 2].scatter(a, d, c=colors, alpha=0.5)
axarr[0, 2].set_title('SL     PW')

axarr[1, 0].scatter(b, c, c=colors, alpha=0.5)
axarr[1, 0].set_title('SW     PL')
axarr[1, 1].scatter(b, d, c=colors, alpha=0.5)
axarr[1, 1].set_title('SW     PW')
axarr[1, 2].scatter(c, d, c=colors, alpha=0.5)
axarr[1, 2].set_title('PL     SW')

f1.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
plt.show()


f2 = plt.figure()
ax = f2.add_subplot(221, projection='3d')
ax.scatter(a, b, c, c=colors, alpha=0.5 ,marker='o')
ax.set_title('SL SW PL')

ax = f2.add_subplot(222, projection='3d')
ax.scatter(a, b, d, c=colors, alpha=0.5 ,marker='o')
ax.set_title('SL SW PW')

ax = f2.add_subplot(223, projection='3d')
ax.scatter(a, c, d, c=colors, alpha=0.5 ,marker='o')
ax.set_title('SL PL PW')

ax = f2.add_subplot(224, projection='3d')
ax.scatter(b, c, d, c=colors, alpha=0.5 ,marker='o')
ax.set_title('SW PL PW')

plt.show()
