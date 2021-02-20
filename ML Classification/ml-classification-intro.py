#!/usr/bin/env python
# coding: utf-8

# Machine Learning Classification / ml-classification-intro
# Gourav Siddhad
# 31-Mar-2019

# # Logistic Regression Example
# #### Source Code Link: https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac

# In[ ]:

import math
import random
import csv
import numpy as np
from sklearn import datasets
import pandas as pd

# In[ ]:

iris = pd.read_csv('iris.csv')
iris.head()

# In[ ]:

iris = np.array(iris)
np.random.shuffle(iris)

iris[:, 4] = (iris[:, 4] != 0)*1
train = iris[:120]
test = iris[120:]

# In[ ]:


class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])
        print("Intial Weights:\t", self.theta)
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)

            if(i % 5000 == 0):

                print("z: \t", z[:5])
                print("h: \t", h[:5])
                print("Gradient: \t", gradient)
                print("Updated Weigths:\t", self.theta)
                print('loss: \t', loss)

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()

# In[ ]:


model = LogisticRegression(lr=0.1, num_iter=30000)

# In[ ]:

x = train[:, 0:4]
y = train[:, 4]
model.fit(x, y)

# In[ ]:

preds = model.predict(test[:, 0:4])
tp, fp, tn, fn = 0, 0, 0, 0
for i in range(len(preds)):
    if preds[i] == 1 and test[i, 4] == 1:
        tp = tp+1
    if preds[i] == 1 and test[i, 4] == 0:
        fp = fp+1
    if preds[i] == 0 and test[i, 4] == 0:
        tn = tn+1
    if preds[i] == 0 and test[i, 4] == 1:
        fn = fn+1

print('Precision:\t', tp/(tp+fp))
print('Recall:\t', tp/(tp+fn))

# # Gaussian Naive Bayes Example
# #### Source Code Link: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

# In[ ]:


# ### Separation of instances based on class type.
# In[ ]:


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    # print(separated)
    return separated

# In[ ]:


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute))
                 for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

# ### Class wise calculation of $\mu\ and\ \sigma$ for each Attribute.
# In[ ]:


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    print(summaries)
    return summaries

# ### Calculating class probability using class wise $\mu\ and\ \sigma$
# <br />
# <font size='4' color='#FF821'>$ f(c| x,\ \mu, \sigma^2) = \frac{\exp{-\frac{{(x-\mu)}^2}{2\sigma^2}}}{\sqrt{2 \pi \sigma^2}}$</font>
# In[ ]:


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

# In[ ]:


def predict(summaries, inputVector):

    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

# In[ ]:


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

# In[ ]:


dataset = np.array(pd.read_csv('iris.csv'))

np.random.shuffle(dataset)

dataset[:, 4] = (dataset[:, 4] != 0)*1
trainingSet = dataset[:120]
testSet = dataset[120:]

summaries = summarizeByClass(trainingSet)
predictions = getPredictions(summaries, testSet)
accuracy = getAccuracy(testSet, predictions)

print('Accuracy:\t', accuracy)
