#!/usr/bin/env python
# coding: utf-8

# Machine Learning / ml-intro.py
# Gourav Siddhad
# 02-Mar-2019

# Introduction to Linear Regression

import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

import sklearn as sk
from sklearn import linear_model, metrics, preprocessing
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

N = 7
x = np.array([1,2,3,4,5,6,7])
y = np.array([1,2,3,4,5,6,7])
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii
plt.scatter(x, y, c='red', s=area[1], alpha=0.5)
plt.show()

lr1 = linear_model.LinearRegression()
lr1.fit(x.reshape(7,1),y.reshape(7,1));

m1 = lr1.coef_[0]  
c1 = lr1.intercept_
print(' y_pred = {0} * x + {1}'.format(m1, c1))


data1=pd.read_csv("hitterdata.csv")
print('Baseball Players\' dataset: HITTER' )
print(data1)
print('shape of the data')
print(data1.shape)

instance_count,feature_count = data1.shape

print(data1.info()) # meta-data about dataset
#data1.

num_data = np.array(data1)
y = num_data[:,19] # extracting target feature
print(y.shape)

fig, ax = plt.subplots(nrows=8, ncols=2, figsize=(10,30))
label =['AtBat'  ,'Hits', 'HmRun', 'Runs', 'RBI', 'Walks','Years', 'CAtBat', 'CHits', 'CHmRun','CRuns', 'CRBI', 'CWalks', 'League', 'Division', 'PutOuts','Assists', 'Errors' ]      

index = [1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18]
m=0
for i in range(8):
    ax[i,0].scatter(num_data[:,index[m]], y, marker='x', color='b')
   
    ax[i,0].set_xlabel(label[m])
    ax[i,0].set_ylabel('Salary')
    ax[i,0].set_title(label[m]+' vs Salary')
    m+=1
        
    ax[i,1].scatter(num_data[:,index[m]], y, marker='o', color='k')
    ax[i,1].set_xlabel(label[m])
    ax[i,1].set_ylabel('Salary')
    ax[i,1].set_title(label[m]+' vs Salary')
    
    m+=1
plt.tight_layout()
plt.show()

# fill up the missing values with the zero
# also try with mean value

y = num_data[:,[19]]
sum =0
for j in range(322):
    if not math.isnan(y[j]):
        sum = sum + y[j]
mean = sum/instance_count
print(int(mean))

for j in range(322):
    if math.isnan(y[j]):     
        y[j]=0            
print(y)

d = np.delete(num_data,[0,14,15,19,20],1)
print(d)

for j in range(16):
    mean = np.mean(d[:,j])
    max = np.max(d[:,j])
    min = np.min(d[:,j])
    std = np.std(d[:,j])
    for i in range(322):
        d[i,j] = (d[i,j]-mean)/(std)
print(d)

x_train1, x_test1, y_train1, y_test1 = train_test_split(d, y, test_size = .3) # slpitting data in to training and testing sets

lr1 = linear_model.LinearRegression() # instantiation of Linear Regression model
lr1.fit(x_train1,y_train1); # learning 

y_pred1 = lr1.predict(x_test1)  # tesing the model
mse1 = sk.metrics.mean_squared_error(y_test1, y_pred1)
print("MSE :",mse1)

m1 = lr1.coef_[0]  
c1 = lr1.intercept_
print(' y_pred = {0} * x + {1}'.format(m1, c1)) # equation of line


# Principal Component Analysis

# url='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';

df = pd.read_csv('iris.csv',sep=',')
df.tail()

X = df.iloc[:,0:4]
Y = df.iloc[:,4]

# Covariance Matrix Calculation:

# step 1 :
X_std = StandardScaler().fit_transform(X) 

# print(X_std)
mean_vec = np.mean(X_std, axis=0)
print(mean_vec)
# print(np.shape(X_std.T))

cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

# Eigen Value and Eigen Vector Calculation
# eigenvalue decomposition

cov_mat = np.cov(X_std.T)
print('Covariance Matrix')
print(cov_mat)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# Sorting Eigen Values 

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

print("----pairs-----")
print(eig_pairs)
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

print("----pairs after sorting-----")
print(eig_pairs)

tot = sum(eig_vals)
print('sum of eig_vals:',tot)
print('% in total sum')

var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print(var_exp)

# Generating New 2-D data using 1st and 2nd best feature direction.
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
print('Matrix W:\n', matrix_w)

# 3 - Projection Onto the New Feature Space
X_pca = X_std.dot(matrix_w)
X_pca

# Demo of the Affect of PCA on the Predictions...

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = .3)

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_test)
print(classification_report(y_test, y_pred))

X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size = .3)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
print(classification_report(y_test, y_pred))
