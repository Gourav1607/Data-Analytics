#!/usr/bin/env python
# coding: utf-8

# Text Processing / textpp-intro.py
# Gourav Siddhad
# 25-Jan-2019

input_str = 'Statistics is one of the main Building Block of Data Science.'
lower = input_str.lower()
print(lower)

upper = input_str.upper()
print(upper)

import nltk as nl

#sentense tokenization
import nltk as nl
import random

example_text = 'a real movie , about real people. Movie gives us a rare glimpse into a culture most of us don\'t know .'
print(nl.tokenize.sent_tokenize(example_text))

#word tokenization
example = 'Movie gives us a rare glimpse into a culture most of us don\'t know .'

tokenized_words = nl.tokenize.word_tokenize(example)
print(tokenized_words)

#pos tagging
tagged_words = nl.tag.pos_tag(tokenized_words)
print(tagged_words)

# removal of stop words

stop_words = set(nl.corpus.stopwords.words('english'))
word_tokens = nl.tokenize.word_tokenize(example)
filtered_words = []

for w in word_tokens:
    if w not in stop_words:
        filtered_words.append(w)
print('stop words provided by NLTK package----\n')
print(stop_words)
print('\n filtered data after stop word removal \n')
print(filtered_words)

#Chunking
chunked_data = nl.chunk.ne_chunk(tagged_words)
print(chunked_data)
chunked_data.draw()

#punctuation mark removal
p = [',','?','.']
word_tokens = filtered_words

filtered2 = []

for w in word_tokens:
    if w not in p:
        filtered2.append(w)

print(word_tokens)
print(filtered2)

# stemming

ps = nl.stem.PorterStemmer()
set1 =['Movie', 'gives', 'us', 'rare', 'glimpse', 'culture', 'us', "n't", 'know']

for w in set1:
    print(ps.stem(w))

set2 =[ "python","pythoner","pythoning","pythoned"]
set2 =[ "banks","banking","bank"]

print('-------------------')
for w in set2:
    print(ps.stem(w))

#data set loading
positive = open('rt-polarity-pos.txt')
negative = open('rt-polarity-neg.txt')

i=0
while i<5 :
    print(negative.readline())
    i+=1

#print(positive.readlines())

#preprocessing
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
    punctuations = [',','?','.',']','[','}','{','(',')','!','?',':',';','"','\'']
    t2 = []
    for w in w_token:
        if w not in punctuations:
            t2.append(w)
    t3 = remove_stop_words(t2)
    return {word: 1 for word in t3}


positive_data_array = []

# positive review prcocessing
for p_review in positive:
    positive_data_array.append([process_sentence(p_review),'positive'])

negative_data_array = []
i= 0

#negative review processing
for n_review in negative:
        processed = [process_sentence(n_review),'negative']
        negative_data_array.append(processed)

        if(i<5):
            print('review before processing->')            # demo purpose code to see the affect
            print(n_review)
            print('review after processing->')
            print(processed)
            print('-------------------------')
            i+=1

#partition into training and test set

# shuffling
random.shuffle(positive_data_array)
random.shuffle(negative_data_array)

#partitioning
training_set = positive_data_array[:3000]+negative_data_array[:3000]
test_set =positive_data_array[3000:]+negative_data_array[3000:]

#build classifier and test
classifier = nl.NaiveBayesClassifier.train(training_set)

print('Accuracy: ',nl.classify.util.accuracy(classifier,test_set))
print('\n')

TP = 0
TN = 0
FP = 0
FN = 0
for itr in range(len(test_set)):
    out=classifier.classify(test_set[itr][0])
    if out=='positive' and test_set[itr][1]=='positive':
        TP = TP + 1
    if out=='negative' and test_set[itr][1]=='negative':
        TN = TN + 1
    if out=='positive' and test_set[itr][1]=='negative':
        FP = FP + 1
    if out=='negative' and test_set[itr][1]=='positive':
        FN = FN + 1

print('True Positive:',TP,'\t False Positive:',FP)
print('False Negative:',FN,'\t True Negative:',TN)

print('\n')
print('Confusion Matrix:\n')
print('_________________________________________________')
print('\t\t\tActual Output\n')
print('_________________________________________________')
print('\t\t',TP,'\t\t',FP)
print('Test Output -------------------------------------')
print('\t\t',FN,'\t\t',TN)
print('_________________________________________________')
print('\n')
print('Precision:[TP/(TP+FP)]\t',TP/(TP+FP))
print('Recall:[TP/(TP+FN)]\t',TP/(TP+FN))
print('Accuaracy:[(TP+TN)/(TP+TN+FP+FN)]\t',(TP+TN)/(TP+TN+FP+FN))

# classifiy new test instance
print(classifier.classify(process_sentence('very bad movie. I have wasted my money.')))
print(classifier.classify(process_sentence('One of the best movie, I have ever seen.')))
print(classifier.classify(process_sentence('a real movie , about real people , that gives us a rare glimpse into a culture most of us don\'t know .')))

from nltk.corpus import wordnet as wn
import nltk

# Then, we're going to use the term "program" to find synsets like so:
syns = wn.synsets("run")

print(syns)
# An example of a synset:
print(syns[0].name())

# Just the word:
print(syns[0].lemmas()[0].name())

# Definition of that first synset:
print(syns[0].definition())

# Examples of the word in use in sentences:
print(syns[0].examples())
