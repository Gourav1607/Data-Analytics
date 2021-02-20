#!/usr/bin/env python
# coding: utf-8

# Social Network Analysis / sna-intro.py
# Gourav Siddhad
# 01-Mar-2019

# Network Analysis with Python
# networkx and pygraphviz

# Network (or Graph): Representation of connections among the set of items related to a particular domain.
# Node or Vertex
# Edge or Link

# Undirected Graph

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms import bipartite

G = nx.Graph()
G.add_edge('S', 'S4')
G.add_edge('S', 'X')

nx.draw(G, with_labels=True)
plt.show()

# Directed Graph

G1 = nx.DiGraph()
G1.add_edge('Derived Class1', 'Base Class')
G1.add_edge('Derived Class2', 'Base Class', relation='P-Cs')
nx.draw(G1, with_labels=True)
plt.show()
# print(G1['Base Class']['Derived Class2'])
print(G1['Derived Class2']['Base Class'])

# Weighted Graph

G2 = nx.Graph()
G2.add_edge('z', 'x', weight=4)
G2.add_edge('y', 'x', weight=5)
pos = nx.spring_layout(G2)
nx.draw_networkx_edge_labels(G2, pos, edge_labels={(
    'y', 'x'): '5', ('z', 'x'): '6'}, font_color='red')
nx.draw(G2, with_labels=True, edge_color=['red', 'green'])
plt.show()

print(G2.edges(data=True))
print(G2.edges())
print(G2.edges(['y']))
print(G2['x']['y'])
print(G2['y']['x'])

# Multi Graph

G3 = nx.MultiGraph()
G3.add_edge('A', 'B', weight=6, relation='family')
G3.add_edge('A', 'B', weight=18, relation='friend')
G3.add_edge('C', 'B', weight=13, relation='friend')
nx.draw(G3, with_labels=True)
plt.show()

print(G3.edges(data=True))
print(G3.edges())
print(G3['A']['B'])
print(G3['B']['A'])

# Task 01: find the way to visualize the edges clearly...
# Bipartite Graph

st1 = ['1', '2', '3']
st2 = ['11', '22', '33']
G4 = nx.Graph()
G4.add_nodes_from(st1, bipartite=0)
G4.add_nodes_from(st2, bipartite=1)
G4.add_edges_from([('1', '11'), ('1', '22'), ('1', '33'), ('2', '11'),
                   ('2', '22'), ('2', '33'), ('3', '11'), ('3', '22'), ('3', '33')])
pos = nx.bipartite_layout(G4, st1)
nx.draw(G4, pos=pos,)
plt.show()
bipartite.is_bipartite(G4)

# Task: labeling of nodes.....

# Local Clustering Coefficient
# Fraction of pairs of the node’s friends that are friends with each other.
# LCC = (no. of pairs of node's friends who are friends(A)) / (no. of pairs of node's friend(B))
# B = n(n-1)/2

A = np.zeros([50, 50])
for i in range(50):
    for j in range(50):
        A[i, j] = np.round(np.random.random())

G5 = nx.from_numpy_matrix(A)
nx.draw(G5, with_labels=True)
plt.show()

nx.clustering(G5, 1)

# Global Clustering Coefficient
# Simple Average

nx.average_clustering(G5)

# Transtivity
# Ratio of number of triangles and number of “open triads” in a network.

nx.transitivity(G5)

G6 = nx.complete_graph(4)
nx.draw(G6)
plt.show()
nx.transitivity(G6)

# Shortest Path

A = np.zeros([5, 5])
for i in range(5):
    for j in range(5):
        A[i, j] = np.round(np.random.random())

GS = nx.from_numpy_matrix(A)
nx.draw(GS, with_labels=True)
plt.show()

print(GS.node())
nx.shortest_path(GS, 0, 2)

# Breadth First Search Tree

GTmp = nx.bfs_tree(GS, 2)
GTmp.edges()

# ## Diameter
# ### maximum distance between any pair of nodes.

nx.diameter(GS)

# Eccentricity
# Maximum distance for a node, out of the distances from all  the nodes.

nx.eccentricity(GS)

# Radius
# Minimum Eccentricity of the graph.

nx.radius(GS)

# Periphery
# set of nodes having eccentricity equal to the Diameter.

nx.periphery(GS)

# Center
# set of nodes having eccentricity equal to the Diameter.

nx.center(GS)

# Connectivity
# Node Connectivity
# Edge Connectivity
# Minimum Node Cut
# Minimum Edge Cut

A = np.array([[0, 1, 0, 1, 0], [1, 0, 1, 1, 0], [
             1, 0, 0, 0, 1], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
GS1 = nx.from_numpy_matrix(A)
nx.draw(GS1, with_labels=True)
plt.show()

print(GS1.node())
print(nx.node_connectivity(GS1))
print(nx.minimum_node_cut(GS1))
print(nx.edge_connectivity(GS1))
print(nx.minimum_edge_cut(GS1))

# Centrality:
# Indentification of Important nodes.

# Degree Centrality
# C_deg(V) = D_V/(|N|-1)
# D_V : Degree of node V
# N : Number of Nodes in the graph

GK = nx.karate_club_graph()
nx.draw(GK, with_labels=True)
plt.show()
print(nx.degree_centrality(GK))

# Degree Cenrtality: Directed Graph
# Closeness Centrality

nx.closeness_centrality(GK)

# Page Rank Algorithm

GP = nx.DiGraph()
GP.add_edge(1, 2)
GP.add_edge(1, 3)
GP.add_edge(4, 3)
GP.add_edge(3, 5)
nx.draw(GP, with_labels=True)
plt.show()
print(nx.pagerank(GP))
print(nx.pagerank_numpy(GP))
print(nx.pagerank_scipy(GP))

##

data = pd.read_csv('soc-sign-bitcoinalpha.csv', header=None)
data = np.array(data)

source = data[:, 0]
target = data[:, 1]
weight = data[:, 2]

G7 = nx.DiGraph()
for a, b in zip(source, target):
    G7.add_edge(a, b)
nx.draw(G7, with_labels=True)
plt.show()

# Full Graph of facebook_combined

G_fb = nx.read_edgelist("facebook_combined.txt",
                        create_using=nx.Graph(), nodetype=int)

pos = nx.spring_layout(G_fb)
betCent = nx.betweenness_centrality(G_fb, normalized=True, endpoints=True)
node_color = [20000.0 * G_fb.degree(v) for v in G_fb]
node_size = [v * 10000 for v in betCent.values()]
plt.figure(figsize=(20, 20))
nx.draw_networkx(G_fb, pos=pos, with_labels=False,
                 node_color=node_color, node_size=node_size)
plt.axis('off')
