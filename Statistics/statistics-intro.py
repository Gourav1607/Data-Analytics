#!/usr/bin/env python
# coding: utf-8

# Statistics / statistics-intro.py
# Gourav Siddhad
# 11-Jan-2019

import random
import collections
import numpy as np
from matplotlib import pyplot as plt

data = 11*np.random.random(100)

data = list(data.astype(int))

x = range(11)

freq = [data.count(k) for k in range(11)]
freq
index = np.arange(len(x))

plt.bar(index, freq)
plt.xlabel('# wickets ', fontsize=20)
plt.ylabel('# Matches', fontsize=20)
plt.xticks(index, x, fontsize=10, rotation=90)
plt.title('Wickets taken by Zack in ODIs...', fontsize=20)
plt.show()

# # Centrality Measures
# # Mean & Median


def findMean(data):
    return np.sum(data)/len(data)


def findMedian(data):

    n = len(data)
    np.sort(data)
    midpoint = n // 2
    if n % 2 == 1:
        return data[midpoint]
    else:
        lo = midpoint - 1
        hi = midpoint
        return (data[lo] + data[hi]) / 2


data = 10*np.random.random(1000)
print(findMean(data))
print(findMedian(data))

# Dispersion Measures
# Variance


def findVariance(data):
    mean = findMean(data)
    diff = data-mean
    return np.sum(diff**2)/(len(data)-1)


data = 10*np.random.random(10)
findVariance(data)

# Standard Deviation


def findSD(data):
    return np.sqrt(findVariance(data))


data = 10*np.random.random(10)
findSD(data)


# Covariance
# Covariance, the paired analogue of variance. Whereas variance measures how a single variable deviates from its mean, covariance measures how two variables vary in tandem from their means.

N = 50
x = np.random.rand(N)
y = x

x = [1692, 1978, 1884, 2151, 2519]
y = [68, 102, 110, 112, 154]
plt.scatter(x, y, c='r')
plt.show()


def findCovariance(x, y):
    meanx = np.mean(x)
    meany = np.mean(y)
    diffx = x-meanx
    diffy = y-meany
    return np.sum(np.dot(diffx, diffy))/(len(x)-1)


findCovariance(x, y)

# Correlation
# scaled version of covariance, which ranges from -1 to 1. Its not affected by the change in scale of values.


def findCorrelation(x, y):
    return findCovariance(x, y)/(findSD(x)*findSD(y))


findCorrelation(x, y)

# Probability
# Suppose you want to flip a coin. Then you expect not to be able to say whether the next toss would  yield a heads or a tails. You might tell a friend that the odds of getting a heads is equal to to the odds of getting a tails, and that both are 1/2. This intuitive notion of odds is a probability.

# Random Variable
# A random variable is a variable whose possible values are numerical outcomes of a random phenomenon. There are two types of random variables, discrete and continuous.

# Discrete Random Variable
# Can take only a countable number of values as outcome .    Example:-      Coin Toss: {H,T}       Throwing Dice: {1,2,3,4,5,6}dg

# Continuous Random Variable
# A continuous random variable is one which takes an infinite number of possible values. Example:-    predicting the percentage of student in the exam.

# Distributions

# Normal Distribution
# Defined by two parameters: mean & standard deviation.
# The density curve is symmetrical, centered about its mean, with its spread determined by its standard deviation.
# Also called Gaussian Distribution.
# $ f(x| \mu, \sigma^2) = \frac{\exp({-\frac{{(x-\mu)}^2}{2\sigma^2}})}{\sqrt{2 \pi \sigma^2}}  $


def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = np.sqrt(2 * np.pi)
    sigma_square = sigma**2
    return np.exp(- (x-mu) ** 2 / (2 * sigma_square)) / (sqrt_two_pi * sigma)


normal_pdf(0)

xs = [x / 10.0 for x in range(-10, 11)]
print(xs)
nm = [normal_pdf(x, sigma=1) for x in xs]
print(nm)

plt.plot(xs, nm, '-', marker='*')
plt.legend()
plt.title("Normal pdf")
plt.show()

xs = [x / 10.0 for x in range(-100, 100)]
plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-', label='mu=0,sigma=1')
plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '--', label='mu=0,sigma=2')
plt.plot(xs, [normal_pdf(x, sigma=0.5)
              for x in xs], ':', label='mu=0,sigma=0.5')
plt.plot(xs, [normal_pdf(x, mu=-1) for x in xs], '-.', label='mu=-1,sigma=1')
plt.legend()
plt.title("Various Normal pdfs")
plt.show()


# Bernoulli Distribution
# The Bernoulli distribution is a discrete distribution having two possible outcomes labelled by n=0 and n=1 in which n=1 ("success") occurs with probability  p and n=0 ("failure") occurs with probability q=1-p, where 0 < p < 1
# It is the simplest Discrete distribution, which is building block for other distributions like: binomial, geometric etc.

def bernoulli_trial(p):
    d = np.random.random()
    return 1 if d < p else 0


bernoulli_trial(0.5)

# Binomial Distribution


def binomial(n, p):
    return sum(bernoulli_trial(p) for _ in range(n))


binomial(10, 0.5)


def make_hist(p, n, num_points):
    data = [binomial(n, p) for _ in range(num_points)]
    histogram = collections.Counter(data)
    plt.bar([x for x in histogram.keys()], [v for v in histogram.values()])
    plt.title("Binomial Distribution")
    plt.show()


make_hist(0.5, 100, 10000)


def coin_trial():
    heads = 0
    for i in range(100):
        if random.random() <= 0.5:
            heads += 1
    return heads


def simulate(n):
    trials = []
    for i in range(n):
        trials.append(coin_trial())
    return trials


data = simulate(10000)
freq = [data.count(k) for k in range(0, 100)]

rg = range(0, 100)

index = np.arange(len(rg))

plt.bar(index, freq)
plt.xlabel('Frequency', fontsize=20)
plt.ylabel('# of Heads', fontsize=20)
plt.xticks(index, rg, fontsize=1, rotation=45)
plt.title('', fontsize=30)
plt.show()

d1 = [1, 2, 3, 4, 5, 6]
d2 = [1, 2, 3, 4, 5, 6]


def throwDice1():
    return d1[int(6*random.random())]


def throwDice2():
    return d2[int(6*random.random())]


sum = []
for i in range(0, 10000):
    sum.append(throwDice1()+throwDice2())

freq1 = [sum.count(k) for k in range(2, 13)]

rg1 = range(2, 13)

index1 = np.arange(len(rg1))

plt.bar(index1, freq1)
plt.xlabel('Frequency', fontsize=20)
plt.ylabel('# of Heads', fontsize=20)
plt.xticks(index1, rg1, fontsize=1, rotation=45)
plt.title('', fontsize=30)
plt.show()

# Uniform Distribution


def dist_uni(x):

    data = [1 if xd > 20 and xd < 50 else 0 for xd in x]
    return data


x = range(101)
y = dist_uni(x)

index = np.arange(len(x))

plt.bar(index, y)
plt.xlabel('X', fontsize=20)
plt.ylabel('Y', fontsize=20)
plt.xticks(index, x, fontsize=10, rotation=90)
plt.show()

year = ['2001', '2002', '2003', '2004', '2005',
        '2006', '2007', '2008', '2009', '2010']
runs = [400, 400, 400, 400, 400, 400, 400, 400, 400, 400]

index = np.arange(len(year))

plt.bar(index, runs)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Runs scored', fontsize=20)
plt.xticks(index, year, fontsize=10, rotation=90)
plt.title('Number of runs scored by XYZ from 2001 to 2010', fontsize=30)
plt.show()
