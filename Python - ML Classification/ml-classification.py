#!/usr/bin/env python
# coding: utf-8

# Machine Learning Classification / ml-classification
# Gourav Siddhad
# 31-Mar-2019

# In[3]:

# e1. You have "titanic" dataset, which contains information about the passengers.

# Class label
#     survived
#     not survived
# Apply Following Preprocessing steps:
#     Drop the 'PassengerId','Name','Ticket','Cabin','Embarked','Parch' columns.
#     Encode the 'sex' column- male:1, female:0
# Fill the missing values in 'age column as follows:
#     For Pclass = 1, age=40
#     For Pclass = 2, age=32
#     For Pclass = 3, age=25
# Apply GaussianNB Classification.
# Print the performance report.

# In[4]:

from __future__ import print_function

print('Importing Libraries', end='')

import csv
import random
import math
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

print(' - Done')

# In[5]:

print('Reading CSV', end='')
titanic = pd.read_csv('titanic.csv')
print(' - Done')
titanic.head()

# In[6]:

print('Dropping Columns', end='')
titanic = titanic.drop(columns=['PassengerId','Name','Ticket','Cabin','Embarked','Parch'])
print(' - Done')
titanic.head(10)

# In[7]:

print('Replacing Sex Female-0, Male-1', end='')
titanic['Sex'] = titanic['Sex'].map({'female': 0, 'male': 1})
print(' - Done')
print('Replacing Age to 100', end='')
titanic['Age'] = titanic['Age'].replace(np.nan, 100)
print(' - Done')
titanic.head(10)

# In[8]:

print('Replacing Age to 40, 32, 25 after Converting to Array', end='')
tit = np.array(titanic, dtype='float32')
for i in range(len(tit)):
    if tit[i, 3] == 100.0:
        if tit[i, 1] == 1.0:
            tit[i, 3] = 40
        if tit[i, 1] == 2.0:
            tit[i, 3] = 32
        if tit[i, 1] == 3.0:
            tit[i, 3] = 25
print(' - Done')

print('Converting back to DataFrame', end='')
titanic = pd.DataFrame(data=tit, columns=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Fare'])
print(' - Done')
titanic.head(10)

# In[9]:

print('Train Test Split 75-25', end='')
train, test = train_test_split(titanic, test_size=0.25, random_state=42)

train_x = train.drop(columns='Pclass')
train_y = train['Pclass']

test_x = test.drop(columns='Pclass')
test_y = test['Pclass']
print(' - Done')

# In[10]:

print('Training Gaussian Naive Bayes', end='')
gnb = GaussianNB()
gnb.fit(train_x, train_y)
print(' - Done')

# In[11]:

print('Testing Gaussian Naive Bayes', end='')
pred_y = gnb.predict(test_x)
print(' - Done')
print()
print('Classification Report')
print(classification_report(test_y, pred_y))
print()
print('Accuracy Score - ', accuracy_score(test_y, pred_y, normalize = True)*100)

# In[12]:

# e2: Wine Dataset(Classification) is given to you.(wine.csv)
#    Apply PCA and GaussianNB(GNB), Logistic Regression(LR) Classification as follows:
# 2.1: Apply PCA to get the eigen vector and Values.
# # [sort eigen values if not in sorted form]
# 2.2: Start with higest eigen value move upto all the values: generate new dataset.
# 2.3: At each step of 2.2, apply GNB, LR Classification and find the Precision, recall [store it].
# 2.4: Draw the 2 line graphs. [comparing classifier performance]
#      2.4.1 # features vs Precision
#      2.4.2 # features vs Recall

# Add more classifiers like SVM, Simple NN etc.
# Try to write Dynamic Code, which can take list of classfiers as input.

# In[13]:

print('Reading CSV', end='')
wine = pd.read_csv('wine.csv', header=None)
print(' - Done')
wine.head(10)

# In[14]:

print('Generated EigenValues and EigenVectors, Sorted', end='')

# Partitioning Dataset to Data:Class
X = wine.iloc[:,1:14]
Y = wine.iloc[:,0]

# Generating Covariance, EigenValues, EigenVectors
X_std = StandardScaler().fit_transform(X) 
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sorting EigenValues and EigenVectors
eig_pairs.sort()
eig_pairs.reverse()
print(' - Done')

# In[15]:
# Select Classifiers

print('Select Classifiers : ')
print('1. Gaussian Naive Bayes')
print('2. Logistic Regression')
print('3. Support Vector Machine')
print('4. Simple Neural Network')
print('0. Done')

clf = set()
while True:
    option = input('Enter Option - ')
    if int(option) is 0:
        break
    else:
        clf.add(int(option))

print()
print('Selected Classifiers - ')
for cls in clf:
    if cls is 1:
        print('\tGaussian Naive Bayes')
    elif cls is 2:
        print('\tLogistic Regression')
    elif cls is 3:
        print('\tSupport Vector Machine')
    elif cls is 4:
        print('\tNeural Network')

# In[16]:
# 2.2: Start with highest eigen value move upto all the values: generate new dataset.
# 2.3: At each step of 2.2, apply Logistic Regression and find the Precision, recall.

values = tuple()
f1score_lr, precision_lr, recall_lr = [], [], []
f1score_nb, precision_nb, recall_nb = [], [], []
f1score_svm, precision_svm, recall_svm = [], [], []
f1score_nn, precision_nn, recall_nn = [], [], []

print('Performing Model Training and Testing - ', end='')
for i in range(13):    
    print(i, end=' ')
    # Performing PCA
    values = values + (eig_pairs[i][1].reshape(13,1),)
    matrix_w = np.hstack(values)
    X_pca = X_std.dot(matrix_w)

    # Partitioning to Train:Test
    X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size = .3)
    
    if 1 in clf:
        # Apply Gaussian Naive Bayes
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        # Test
        y_pred_nb = gnb.predict(X_test)
        # Saving Scores
        f1score_nb.append(f1_score(y_test, y_pred_nb, average='macro'))
        precision_nb.append(precision_score(y_test, y_pred_nb, average='weighted'))
        recall_nb.append(recall_score(y_test, y_pred_nb, average='macro'))

    if 2 in clf:
        # Apply Logistic Regression
        LogReg = LogisticRegression()
        LogReg.fit(X_train, y_train)
        # Test
        y_pred_lr = LogReg.predict(X_test)
        # Saving Scores
        f1score_lr.append(f1_score(y_test, y_pred_lr, average='macro'))
        precision_lr.append(precision_score(y_test, y_pred_lr, average='weighted'))
        recall_lr.append(recall_score(y_test, y_pred_lr, average='macro'))

    if 3 in clf:
        # Apply SVM
        SupportVM = SVC()
        SupportVM.fit(X_train, y_train) 
        # Test
        y_pred_svm = SupportVM.predict(X_test)
        # Saving Scores
        f1score_svm.append(f1_score(y_test, y_pred_svm, average='macro'))
        precision_svm.append(precision_score(y_test, y_pred_svm, average='weighted'))
        recall_svm.append(recall_score(y_test, y_pred_svm, average='macro'))
    
    if 4 in clf:
        # Apply Neural Network
        nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 12), random_state=1)
        nn.fit(X_train, y_train)
        # Test
        y_pred_nn = nn.predict(X_test)
        # Saving Scores
        f1score_nn.append(f1_score(y_test, y_pred_nn, average='macro'))
        precision_nn.append(precision_score(y_test, y_pred_nn, average='weighted'))
        recall_nn.append(recall_score(y_test, y_pred_nn, average='macro'))

print(' - Done')

# In[17]:
# For Neural Network, different Hidden Layer Config changes the behaviour of the network
# 2.4: Draw the line graph
    # 2.4.1 # features vs Precision
    # 2.4.2 # features vs Recall

index = [x for x in range(1,14)]
fig, axes = plt.subplots(2, 2, figsize=(15,15))

if 1 in clf:
    axes[0, 0].plot(index, f1score_nb, label='Naive B')
    axes[0, 1].plot(index, precision_nb, label='Naive B')
    axes[1, 0].plot(index, recall_nb, label='Naive B')
    axes[1, 1].plot(index, f1score_nb, label='NB F1 Score')
    axes[1, 1].plot(index, precision_nb, label='NB Precision')
    axes[1, 1].plot(index, recall_nb, label='NB Recall')

if 2 in clf:
    axes[0, 0].plot(index, f1score_lr, label='Log Reg')
    axes[0, 1].plot(index, precision_lr, label='Log Reg')
    axes[1, 0].plot(index, recall_lr, label='Log Reg')
    axes[1, 1].plot(index, f1score_lr, label='LR F1 Score')
    axes[1, 1].plot(index, precision_lr, label='LR Precision')
    axes[1, 1].plot(index, recall_lr, label='LR Recall')

if 3 in clf:
    axes[0, 0].plot(index, f1score_svm, label='SVM SVC')
    axes[0, 1].plot(index, precision_svm, label='SVM SVC')
    axes[1, 0].plot(index, recall_svm, label='SVM SVC')
    axes[1, 1].plot(index, f1score_svm, label='SVM F1 Score')
    axes[1, 1].plot(index, precision_svm, label='SVM Precision')
    axes[1, 1].plot(index, recall_svm, label='SVM Recall')

if 4 in clf:
    axes[0, 0].plot(index, f1score_nn, label='Neural Net')
    axes[0, 1].plot(index, precision_nn, label='Neural Net')
    axes[1, 0].plot(index, recall_nn, label='Neural Net')
    axes[1, 1].plot(index, f1score_nn, label='NN F1 Score')
    axes[1, 1].plot(index, precision_nn, label='NN Precision')
    axes[1, 1].plot(index, recall_nn, label='NN Recall')

axes[0, 0].set_title('F1 Accuracy')
axes[0, 0].set_xlabel('# of Features')
axes[0, 0].set_ylabel('Score')
axes[0, 0].legend()
axes[0, 1].set_title('Precision')
axes[0, 1].set_xlabel('# of Features')
axes[0, 1].set_ylabel('Score')
axes[0, 1].legend()
axes[1, 0].set_title('Recall')
axes[1, 0].set_xlabel('# of Features')
axes[1, 0].set_ylabel('Score')
axes[1, 0].legend()
axes[1, 1].set_title('F1Score vs Precision vs Recall')
axes[1, 1].set_xlabel('# of Features')
axes[1, 1].set_ylabel('Score')
axes[1, 1].legend()

plt.savefig('Classification Plot.jpg', dpi=300, pad_inches=0.1)
plt.show()
