#!/usr/bin/env python
# coding: utf-8

# Basics / plot-intro.py
# Gourav Siddhad
# 04-Jan-2019

from matplotlib import pyplot as plt
import numpy as np

x = [19, 45, 56, 67, 78, 123, 234]
y1 = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]
y2 = [30.2, 240.3, 675.9, 1262.5, 3579.6, 7089.7, 10958.3]
plt.plot(x, y1, color='blue', marker='*', linestyle='solid')
plt.plot(x, y2, color='red', marker='o', linestyle='solid')
plt.show()
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()

data = [1, 2, 15, 3, 6, 17, 8, 16, 8, 3, 10, 12, 16, 12, 9]
nparray = np.array(data)

print(nparray.data)
print("Mean--", np.mean(nparray))
print("Median--", np.median(nparray))
print("Standard Div.  --", np.std(nparray))
print("Max--", np.max(nparray))
print("Min--", np.min(nparray))

plt.boxplot(nparray.data)
plt.show()

year = ['2001', '2002', '2003', '2004', '2005',
        '2006', '2007', '2008', '2009', '2010']
runs = [
    941,
    854,
    1595,
    625,
    942,
    509,
    548,
    749,
    1052,
    961
]

index = np.arange(len(year))

plt.bar(index, runs)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Runs scored', fontsize=20)
plt.xticks(index, year, fontsize=10, rotation=90)
plt.title('Number of runs scored by XYZ from 2001 to 2010', fontsize=30)
plt.show()

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()

labels = 'USA', 'India', 'Japan', 'China'
sizes = [30, 15, 45, 10]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels)

plt.show()
