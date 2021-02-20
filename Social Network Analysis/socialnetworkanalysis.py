#!/usr/bin/env python
# coding: utf-8

# Social Network Analysis / socialnetworkanalysis.py
# Gourav Siddhad
# 01-Mar-2019

# Michigen University Course
# Github codes on social network analysis
# Consequences of Simple average and transitivity

# Exercise
# 1   1.1 Generate 100 "Complete graphs" of size 20, 40, .....2000 nodes Randomly.
#     1.2 Calculate Global Clustering Co. (average, transtivity) for each graph.
#     1.3 Plot the line graph to infer the relationships.
#         ( Node count vs GCCavg,Node Count vs GCCtranNode count vs GCCavg,Node Count vs GCCtran ).

# 2 Visit the site: https://snap.stanford.edu/data. Select data of your choice.
#   Calculate all the parameters aplicable on that graph.

import matplotlib.pyplot as plt
import networkx as nx
from random import shuffle
import pandas as pd
import numpy as np
import csv
print('Importing Libraries', end='')


print(' - Done')

# 1.1 Generate 100 "random graphs" of size 20, 40, .....2000 nodes Randomly.
# 1.2 Calculate Global Clustering Co. (average, transtivity) for each graph.

gcc_avg, gcc_trn = [], []

# Change this Value from 100 to 25 Graphs
ngraphs = 100
# For now, this is running for 100 Graphs

nodes = []
for x in range(1, ngraphs+1):
    nodes.append((20*x) % 2020)

# If need nodes to be random for each graph, else they are sequential 20, 40, ...
# shuffle(nodes)

print(ngraphs, ' Graphs with Number of Nodes : ')
print(nodes)

print('\nGenerating and Calculating Graphs - ', end='')
for i in range(0, ngraphs):  # Number of Graphs
    print(i+1, end=' ')
    A = np.zeros([nodes[i], nodes[i]])

    # Randomly Initialize each Graph's Matrix
    for j in range(0, nodes[i]):
        for k in range(0, nodes[i]):
            A[j, k] = np.round(np.random.random())

    # Generate Graph from Matrix
    myGraph = nx.from_numpy_matrix(A)

    # Calculate Values
    gcc_avg.append(nx.average_clustering(myGraph))
    gcc_trn.append(nx.transitivity(myGraph))

print('\nGenerating and Calculating Graphs - Done')

# UnComment to See all the values
# print()
# print(gcc_avg)
# print(gcc_trn)

# 1.3 Plot the line graph to infer the relationships.
#     ( Node count vs GCCavg,Node Count vs GCCtranNode count vs GCCavg,Node Count vs GCCtran ).

plt.plot(nodes[0:ngraphs+1], gcc_avg, color='green')
plt.title("GCC Average")
plt.xlabel('Nodes')
plt.ylabel('Score')
plt.savefig('C:\\Users\\IVELab1\\Documents\\CS533L\\GCC_Average.png')
plt.show()

plt.plot(nodes[0:ngraphs+1], gcc_trn, color='blue')
plt.title("GCC Transitivity")
plt.xlabel('Nodes')
plt.ylabel('Score')
plt.savefig('C:\\Users\\IVELab1\\Documents\\CS533L\\GCC_Transitivity.png')
plt.show()

# Q2 Consequence of Simple Average and Transitivity
# Transitivity, expresses how interconnected a graph is in terms of a ratio of actual over possible connections.
# Measurements like transitivity concern likelihoods rather than certainties

diff = []
for i in range(0, ngraphs):
    diff.append(gcc_avg[i] - gcc_trn[i])

plt.plot(nodes[0:ngraphs+1], diff, color='blue')
plt.title("Difference between Average and Transitivity")
plt.xlabel('Nodes')
plt.ylabel('Difference')
plt.savefig('C:\\Users\\IVELab1\\Documents\\CS533L\\Diff_Avg_Trans.png')
plt.show()

# Writing Values in CSV to read Later

with open('Avg_Trans.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerow(gcc_avg)
    writer.writerow(gcc_trn)

writeFile.close()
