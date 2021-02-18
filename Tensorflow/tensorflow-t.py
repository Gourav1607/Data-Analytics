#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Tensorflow / tensorflow-t
# Gourav Siddhad
# 06-Apr-2019

# In[2]:

# The Project Directory is as follows
# Root / Training / ClassFolders / Images
# Root / Testing  / ClassFolders / Images
# Root / *.ipynb

# In[3]:

from __future__ import print_function

print('Importing Libraries', end='')

import os
import cv2
import csv
import numpy as np
import pandas as pd
import tensorflow as tf

from skimage import io
from skimage.io import imread

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

print(' - Done')

# In[4]:

# # Dataset of "traffic sign images" is given to you. [train_data zip file, test_data zip file]
# # Apply Logistic Regression to classify images as follows:
# # 1.1 Read all the Data from the folders.
# # 1.2 Resize images to 28X28.
# # 1.3 Convert RGB to Gray scale Image.
# # 1.4 Create NN architecture as follows:
#        Input layer: 784
#        First Hidden Layer: 256
#        Second Hidden Layer: 128
#        Third Hidden Layer: 256
#        OutPut Layer: # of classes
# # 1.5 Perform Training.
# # 1.6 Test your model.

# In[5]:

train_path = 'Training/'
test_path = 'Testing/'

print('Train Folder - ', train_path)
print('Test Folder  - ', test_path)

train_folders = []
test_folders = []

print()
print('Reading Path Folders')
print('Train', end='')
for folder in os.listdir(train_path):
    if os.path.isdir(train_path + folder):
        train_folders.append(folder)
print(' - Done')

print('Test', end='')
for folder in os.listdir(test_path):
    if os.path.isdir(test_path + folder):
        test_folders.append(folder)
print(' - Done')
print()
print('Train Folders - ', len(train_folders))
print('Test Folders  - ', len(test_folders))

# In[25]:

def read_images(folders, path):
    images = []
    labels = []
    columns = ['Filename', 'Width', 'Height', 'X1', 'Y1', 'X2', 'Y2', 'ClassID']

    i = 0
    for folder in folders:
        print(i, end=' ')
        tempdf = pd.DataFrame(columns=columns)

        # Read CSV
        for image in os.listdir(path + folder):
            if image.endswith('.csv'):   
                tempdf = pd.read_csv(path + folder + '/' + image, delimiter=';')
                tempdf.columns = ['Filename', 'Width', 'Height', 'X1', 'Y1', 'X2', 'Y2', 'ClassID']
                break
        
        # Read Images
        for image in os.listdir(path + folder):
            if not image.endswith('.csv'):
                tempdf2 = tempdf.loc[tempdf['Filename'] == str(image)]
                # Use these details to crop, resize and label
                img = cv2.imread(path + folder + '/' + image)
                crop_img = img[tempdf2.iloc[0]['Y1']:tempdf2.iloc[0]['Y2'], tempdf2.iloc[0]['X1']:tempdf2.iloc[0]['X2']]
                resized = cv2.resize(crop_img, (28, 28))
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                images.append(np.array(gray)/255.0)
                labels.append(tempdf2.iloc[0]['ClassID'])
        i += 1
        
    return np.array(images), np.array(labels)

# In[26]:

print('Reading Train Images')
train_images, train_labels = read_images(train_folders, train_path)
print(' - Done')

print('Reading Test Images')
test_images, test_labels = read_images(test_folders, test_path)
print(' - Done')

print()
print('Train : Images - ', len(train_images), 'Labels - ', len(train_labels))
print('Test  : Images - ', len(test_images), 'Labels - ', len(test_labels))

# In[27]:

print('Converting to Numpy Array')
print('Train', end='')
train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)
print(' - Done')
print('Test', end='')
test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)
print(' - Done')
print()
print('Train Images - ', train_images.shape)
print('Test Images  - ', test_images.shape)
print('Train Labels - ', train_labels.shape)
print('Test Labels  - ', test_labels.shape)

# In[114]:

# Network Configuration

# Input Layer
n_input = 784
# Hidden Layers
n_hidden_1, n_hidden_2, n_hidden_3 = 256, 128, 256
# Output Layer
n_classes = len(train_folders) # 62
# Parameters
learning_rate = 0.002

print('Network Configuration')
print('Input  - ', n_input)
print('Hidden - ', n_hidden_1, n_hidden_2, n_hidden_3)
print('Output - ', n_classes)
print('L Rate - ', learning_rate)

# In[115]:

print('Creating Network', end='')
# Defining Network

# Input Layer
x = tf.placeholder(tf.float32, [None, n_input], name='features')
y = tf.placeholder(tf.float32, [None, n_classes], name='labels')

# Weight Layers
# W_l1 = tf.Variable(tf.ones([n_input, n_hidden_1]))
# b_l1 = tf.Variable(tf.zeros([n_hidden_1]))
# W_l2 = tf.Variable(tf.ones([n_hidden_1, n_hidden_2]))
# b_l2 = tf.Variable(tf.zeros([n_hidden_2]))
# W_l3 = tf.Variable(tf.ones([n_hidden_2, n_hidden_3]))
# b_l3 = tf.Variable(tf.zeros([n_hidden_3]))
# W_lout = tf.Variable(tf.ones([n_hidden_3, n_classes]))
# b_lout = tf.Variable(tf.zeros([n_classes]))

# Hidden Layers
# layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W_l1), b_l1))
# layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, W_l2), b_l2))
# layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, W_l3), b_l3))

# Output Layer
# layer_out = tf.nn.softmax(tf.add(tf.matmul(layer_3, W_lout), b_lout))

initializer = tf.contrib.layers.xavier_initializer()

# Hidden Layers
layer_1 = tf.layers.dense(x, n_hidden_1, activation=tf.nn.sigmoid, kernel_initializer=initializer)
layer_2 = tf.layers.dense(layer_1, n_hidden_2, activation=tf.nn.sigmoid, kernel_initializer=initializer)
layer_3 = tf.layers.dense(layer_2, n_hidden_3, activation=tf.nn.sigmoid, kernel_initializer=initializer)
layer_out = tf.layers.dense(layer_3, n_classes, activation=tf.nn.softmax)

# Minimize Cost (Error)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=layer_out, labels=y))

# Optimizer - Adam
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

predicted = tf.nn.sigmoid(layer_out)
correct_pred = tf.equal(tf.round(predicted), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print(' - Done')

# In[116]:

# Create Mini Batches to feed the Network
def nextMiniBatch(images, labels, index, batch_size):
    imgs = []
    for img in images[index*batch_size : (index+1)*batch_size]:
        temp = np.reshape(img, (784))
        imgs.append(temp)
        
    labs = []
    for label in labels[index*batch_size : (index+1)*batch_size]:
        temp = np.zeros(62)
        temp[label] = 1
        labs.append(temp)
        
    return np.array(imgs), np.array(labs)

# If either Cost or Accuracy is not Improving, Then Shut Down the model
def checkPatience(cost, accuracy, patience):
    if len(cost) < patience:
        return False
    for i in range(1, patience-1):
        if (cost[-i] != cost[-i-1]) or (accuracy[-i] != accuracy[-i-1]):
            return False
    return True

# In[117]:

# Training Parameters
training_epochs = 1000
batch_size = 256
display_step = 1 # Print cost and accuracy after every this number of epochs
patience = 5 # Consistent same results for this number of epochs will stop training

# In[118]:

# To Save Network History
epoch_list = []
cost_list = []
acc_list = []

# Initialize the Variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)
    print('Training Model - ')
    # For each Epoch
    for epoch in range(training_epochs):
        avg_cost, avg_acc = 0., 0.
        total_batch = len(train_images)//batch_size
        print("Epoch :", '%04d' % (epoch+1), end=' [')
        temp_acc= []
        # For each Batch
        for i in range(total_batch):
            print('#', end='')
            batch_xs, batch_ys = nextMiniBatch(train_images, train_labels, i, batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += loss / total_batch
            # Compute average accuracy
            avg_acc += acc / total_batch
            temp_acc.append(acc)
            
        print(']', end='')
        
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print(" Cost :", "{:.8f}".format(avg_cost), "| Acc :", "{:.8f}".format(avg_acc))
#         print(temp_acc)
        epoch_list.append(epoch + 1)
        cost_list.append(avg_cost)
        acc_list.append(avg_acc)
        
        if checkPatience(cost_list, acc_list, patience) is True:
            print('Accuracy and Cost not Improving - Saving Model')
            break

    print()
    print('Testing Model - ')
    
    # Test model
    pred_y = tf.equal(tf.argmax(layer_out, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(pred_y, tf.float32))
    test_x, test_y = nextMiniBatch(test_images, test_labels, 0, len(test_images))
    print("Test Accuracy :", accuracy.eval({x: test_x, y: test_y}))

# In[119]:

# Plot training & test, cost and accuracy values
plt.plot(cost_list, label='Cost')
plt.title('Cost - NN')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.legend(['Cost'], loc='upper left')
plt.savefig('NN_Cost.png', dpi=300, pad_inches=0.1)
plt.show()

# Plot training & test, cost and accuracy values
plt.plot(acc_list, label='Accuracy')
plt.title('Accuracy - NN')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy'], loc='upper left')
plt.savefig('NN_Acc.png', dpi=300, pad_inches=0.1)
plt.show()

# In[ ]:

# Show onw figure for each traffic sign
plt.figure(figsize=(25, 12))
plt.subplots_adjust(hspace = .1, wspace=.1)
for i in range(0, n_classes):
    index = np.where(train_labels==i)[0][0]
    image = train_images[index]
    plt.subplot(7, 10, i + 1), plt.imshow(image)
    plt.xticks([]), plt.yticks([])
plt.savefig('exploratory.jpg', dpi=300, pad_inches=0.1)
