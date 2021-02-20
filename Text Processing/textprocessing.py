#!/usr/bin/env python
# coding: utf-8

# Text Processing / textprocessing.py
# Gourav Siddhad
# 25-Jan-2019

# e1. You two text files as follows:
# apple_comp_data.txt: contains some sentences about apple company.
# apple_fruit_data.txt: contains some sentences about apple fruit.
# 1.1 Load the text data from the files.
# 1.2 Preprocess the Data.
# 1.3 Apply NB classifier.
# 1.4 Print the Performance.

# e2. You have 5 text documents in zipped format (docs.zip).
# 1.2 Load text data from the files.
# 1.2 Preprocess the data.

# e3. Create document-term matrix.[don't use inbuilt utility]
# Document-Term Matrix: A document-term matrix is a mathematical matrix that describes the frequency of terms
# that occur in a collection of documents. In a document-term matrix, rows correspond to documents in the
# collection and columns correspond to terms.

import nltk as nl
import random
import io
import pandas as pd
import numpy as np

# 1.1 Load the text data from the files.
apple_comp = open('apple_comp_data.txt', encoding="utf-8")
apple_fruit = open('apple_fruit_data.txt', encoding="utf-8")

# 1.2 Preprocess the Data.


def remove_stop_words(w_token):
    stop_words = set(nl.corpus.stopwords.words('english'))
    filtered_words = []
    ps = nl.stem.PorterStemmer()
    for tmp_word in w_token:
        if tmp_word not in stop_words:
            filtered_words.append(tmp_word)
    return filtered_words


def process_sentence(s):
    w_token = nl.tokenize.word_tokenize(s)
    punctuations = [',', '?', '.', ']', '[', '}',
                    '{', '(', ')', '!', '?', ':', ';', '"', '\'', '\'', '\"', '\’']
    t2 = []
    for w in w_token:
        if w not in punctuations:
            t2.append(w)
    t3 = remove_stop_words(t2)
    return {word: 1 for word in t3}


# Company Data Processing
comp_data_array = []
for compinfo in apple_comp:
    processed = [process_sentence(compinfo), 'company']
    comp_data_array.append(processed)

# Fruit data Processing
fruit_data_array = []
for fruitinfo in apple_fruit:
    processed = [process_sentence(fruitinfo), 'fruit']
    fruit_data_array.append(processed)

apple_comp.close()
apple_fruit.close()

# Shuffle Data
random.shuffle(comp_data_array)
random.shuffle(fruit_data_array)

# Partitioning
training_set = comp_data_array[:117] + fruit_data_array[:45]
test_set = comp_data_array[117:] + fruit_data_array[45:]

# 1.3 Apply NB classifier.

# Build Classifier and Test
classifier = nl.NaiveBayesClassifier.train(training_set)
print('Accuracy: ', nl.classify.util.accuracy(classifier, test_set))

# 1.4 Print the Performance.

TP = 0
TN = 0
FP = 0
FN = 0

for itr in range(len(test_set)):
    out = classifier.classify(test_set[itr][0])
    if out == 'fruit' and test_set[itr][1] == 'fruit':
        TP = TP + 1
    if out == 'company' and test_set[itr][1] == 'company':
        TN = TN + 1
    if out == 'fruit' and test_set[itr][1] == 'company':
        FP = FP + 1
    if out == 'company' and test_set[itr][1] == 'fruit':
        FN = FN + 1

print('True Positive:', TP, '\t False Positive:', FP)
print('False Negative:', FN, '\t True Negative:', TN)
print()
print('Confusion Matrix:')
print('_________________________________________________')
print('\t\t\tActual Output')
print('_________________________________________________')
print('\t\t', TP, '\t\t', FP)
print('Test Output -------------------------------------')
print('\t\t', FN, '\t\t', TN)
print('_________________________________________________')
print()
print('Precision:[TP/(TP+FP)]\t', TP/(TP+FP))
print('Recall:[TP/(TP+FN)]\t', TP/(TP+FN))
print('Accuaracy:[(TP+TN)/(TP+TN+FP+FN)]\t', (TP+TN)/(TP+TN+FP+FN))

# e2. You have 5 text documents in zipped format (docs.zip).
# 1.2 Load text data from the files.
# 1.2 Preprocess the data.

d1 = open('d1.txt', encoding="utf-8")
d2 = open('d2.txt', encoding="utf-8")
d3 = open('d3.txt', encoding="utf-8")
d4 = open('d4.txt', encoding="utf-8")
d5 = open('d5.txt', encoding="utf-8")

# d1 to d5 : Data PreProcessing
d1array, d2array, d3array, d4array, d5array = [], [], [], [], []
for sample in d1:
    d1array.append(process_sentence(sample))
for sample in d2:
    d2array.append(process_sentence(sample))
for sample in d3:
    d3array.append(process_sentence(sample))
for sample in d4:
    d4array.append(process_sentence(sample))
for sample in d5:
    d5array.append(process_sentence(sample))

print('Sample of Preprocessed data')
print(d1array)

# e3. Create document-term matrix.[don't use inbuilt utility]
# Document-Term Matrix: A document-term matrix is a mathematical matrix that describes the frequency of terms
# that occur in a collection of documents. In a document-term matrix, rows correspond to documents in the
# collection and columns correspond to terms.

rows = []
rows.extend(d1array)
rows.extend(d2array)
rows.extend(d3array)
rows.extend(d4array)
rows.extend(d5array)
mlist = []
for row in rows:
    for x in row:
        mlist.append(x.lower())

rows = list(set(mlist))
# print(columns)
columns = ['d1', 'd2', 'd3', 'd4', 'd5']
# print(rows)

docmatrix = pd.DataFrame(index=rows, columns=columns)
docmatrix = docmatrix.fillna(0)

for x in d1array:
    for p in x:
        docmatrix['d1'][p.lower()] += 1
for x in d2array:
    for p in x:
        docmatrix['d2'][p.lower()] += 1
for x in d3array:
    for p in x:
        docmatrix['d3'][p.lower()] += 1
for x in d4array:
    for p in x:
        docmatrix['d4'][p.lower()] += 1
for x in d5array:
    for p in x:
        docmatrix['d5'][p.lower()] += 1

docmatrix
