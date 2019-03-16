#!/usr/bin/env python
# coding: utf-8

# Machine Learning / machinelearning.py
# Gourav Siddhad
# 02-Mar-2019

from __future__ import print_function

print('Importing Libraries', end='')

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import sklearn as sk
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report

import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score

print(' - Done')

# 1: Concrete analysis dataset is given to you.(Concrete_data.csv)
# Apply Linear Regression on this dataset as follows
    # 1.1: Simple LR with each input feature one-by-one and Visualize.
    # 1.2: Multiple LR by considering all features at once.
    # 1.3: Draw the performance Bar plot(feature selected vs MSE)

# Reading DataSet to DataFrame
con_data = pd.read_csv("Concrete_Data.csv")

# print('Shape of the data')
# print(con_data.shape)

# print('Concrete DataSet' )
# print(con_data)
# print()
print('Meta Data')
print(con_data.info())

instance_count, feature_count = con_data.shape

num_data = np.array(con_data)
y = num_data[:,8] # extracting target feature

##
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(20,20))

print('One Feature vs Strength')
label =['Cement Quantity', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']
index = [1,2,3,4,5,6,7,8]

m=0
for i in range(4):
    ax[i,0].scatter(num_data[:,index[m]], y, marker='x', color='b')
   
    ax[i,0].set_xlabel(label[m])
    ax[i,0].set_ylabel('Concrete Compressive Strength')
    ax[i,0].set_title(label[m]+' vs Strength')
    m+=1
        
    ax[i,1].scatter(num_data[:,index[m]], y, marker='o', color='k')
    ax[i,1].set_xlabel(label[m])
    ax[i,1].set_ylabel('Concrete Compressive Strength')
    ax[i,1].set_title(label[m]+' vs Strength')
    m+=1
    
plt.tight_layout()
plt.show()

d = num_data
for i in range(0, 8):
    x = num_data[:,i]
    for j in range(8):
        mean = np.mean(d[:,j])
        max = np.max(d[:,j])
        min = np.min(d[:,j])
        std = np.std(d[:,j])
        for i in range(num_data.shape[0]):
            d[i,j] = (d[i,j]-mean)/(std)

# 1.1: Simple LR with each input feature one-by-one and Visualize.

equation1, mse1 = [], []
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(20,20))

# Loop, for each feature
for i in range(0,8):
    # Partitioning to Train:Test
    x_train, x_test, y_train, y_test = train_test_split(d[:,i].reshape(-1,1), y, test_size = .3)
    
    # Creating and Training Model
    lin_model = linear_model.LinearRegression()
    lin_model.fit(x_train, y_train)
    
    # Testing Model
    y_pred = lin_model.predict(x_test)
    
    # Saving Results
    equation1.append([lin_model.coef_[0], lin_model.intercept_])
    mse1.append(sk.metrics.mean_squared_error(y_test, y_pred))
    
    eqline = []
    for j in range(0,100):
        eqline.append(j*lin_model.coef_[0] + lin_model.intercept_)
        
    if i%2 is 0:
        ax[int(i/2),0].scatter(x_test, y_test, color='blue')
        ax[int(i/2),0].plot(x_test, y_pred, color='red', linewidth=3)
        ax[int(i/2),0].set_title('Feature {}'.format(i+1))
    else:
        ax[int(i/2),1].scatter(x_test, y_test, color='blue')
        ax[int(i/2),1].plot(x_test, y_pred, color='red', linewidth=3)
        ax[int(i/2),1].set_title('Feature {}'.format(i+1))

    
print('For 1 Feature at a Time')
for i in range(0,8):
    print(equation1[i], ' - ', mse1[i])
    
plt.show()

# 1.2: Multiple LR by considering all features at once.

equation_all, mse_all = [], []

# Partitioning to Train:Test
x_train, x_test, y_train, y_test = train_test_split(d[:,0:7], y, test_size = .3)

# Creating and Training Model
lin_model = linear_model.LinearRegression()
lin_model.fit(x_train, y_train)

# Testing Model
y_pred = lin_model.predict(x_test)

# Saving Scores
equation_all.append([lin_model.coef_[0], lin_model.intercept_])
mse_all.append(sk.metrics.mean_squared_error(y_test, y_pred))
print('For All Feature at a Time')
print(equation_all, ' - ', mse_all)

# 1.3 Draw the performance Bar plot(feature selected vs MSE)
# All MSE's

mse = mse1 + mse_all
plt.bar(np.arange(len(mse)), mse, align='center', alpha=0.8, color='bbbbbbbbg')
plt.show()

# 2: Wine Dataset(Classification) is given to you.(wine.csv)
# Apply PCA and Logistic Regression as follows:
    # 2.1: Apply All steps of PCA to get the eigen vector and Values (sorted).
    # 2.2: Start with highest eigen value move upto all the values: generate new dataset.
    # 2.3: At each step of 2.2, apply Logistic Regression and find the Precision, recall.
    # 2.4: Draw the line graph
        # 2.4.1 # features vs Precision
        # 2.4.2 # features vs Recall

# Reading DataSet to a DataFrame
df = pd.read_csv('wine.csv', sep=',', header=None)
df.head()

# 2.1: Apply All steps of PCA to get the eigen vector and Values (sorted).

print('Generated EigenValues and EigenVectors, Sorted', end='')

# Partitioning Dataset to Data:Class
X = df.iloc[:,1:14]
Y = df.iloc[:,0]

# Generating Covariance, EigenValues, EigenVectors
X_std = StandardScaler().fit_transform(X) 
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sorting EigenValues and EigenVectors
eig_pairs.sort()
eig_pairs.reverse()
print(' - Done')

# 2.2: Start with highest eigen value move upto all the values: generate new dataset.
# 2.3: At each step of 2.2, apply Logistic Regression and find the Precision, recall.

values = tuple()
f1score, precision, recall = [], [], []

for i in range(13):    
    # Performing PCA
    values = values + (eig_pairs[i][1].reshape(13,1),)
    matrix_w = np.hstack(values)
    X_pca = X_std.dot(matrix_w)

    # Partitioning to Train:Test
    X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size = .3)
    LogReg = LogisticRegression()
    LogReg.fit(X_train, y_train)
    
    # Testing Model
    y_pred = LogReg.predict(X_test)
    
    # Saving Scores
    f1score.append(f1_score(y_test, y_pred, average='macro'))
    precision.append(precision_score(y_test, y_pred, average='weighted'))
    recall.append(recall_score(y_test, y_pred, average='macro'))

# Printing Scores
print('F1 Score - ')
print(f1score)
print('Precision - ')
print(precision)
print('Recall - ')
print(recall)

# 2.4: Draw the line graph
    # 2.4.1 # features vs Precision
    # 2.4.2 # features vs Recall

index = [x for x in range(0,13)]

plt.plot(index, f1score)
plt.title('F1 Accuracy')
plt.xlabel('# of Features')
plt.ylabel('Score')
plt.show()

plt.plot(index, precision)
plt.title('Precision')
plt.xlabel('# of Features')
plt.ylabel('Score')
plt.show()

plt.plot(index, recall)
plt.title('Recall')
plt.xlabel('# of Features')
plt.ylabel('Score')
plt.show()

plt.plot(index, f1score, label='F1 Score')
plt.plot(index, precision, label='Precision')
plt.plot(index, recall, label='Recall')
plt.legend()
plt.title('F1Score vs Precision vs Recall')
plt.xlabel('# of Features')
plt.ylabel('Score')
plt.show()
