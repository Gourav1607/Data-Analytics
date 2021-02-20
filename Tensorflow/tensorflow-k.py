#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Tensorflow / 1811010_Lab_11
# Gourav Siddhad
# 1811010
# 06-Apr-2019

# In[2]:

# The Project Directory is as follows
# Root / Training / ClassFolders / Images
# Root / Testing  / ClassFolders / Images
# Root / *.ipynb

# In[3]:

from __future__ import print_function
import warnings
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import io
from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model, Model
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import csv
import cv2
import os

print('Importing Libraries', end='')


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

# In[6]:


def read_images(folders, path):
    images = []
    labels = []
    columns = ['Filename', 'Width', 'Height',
               'X1', 'Y1', 'X2', 'Y2', 'ClassID']

    i = 0
    for folder in folders:
        print(i, end=' ')
        tempdf = pd.DataFrame(columns=columns)

        # Read CSV
        for image in os.listdir(path + folder):
            if image.endswith('.csv'):
                tempdf = pd.read_csv(
                    path + folder + '/' + image, delimiter=';')
                tempdf.columns = ['Filename', 'Width',
                                  'Height', 'X1', 'Y1', 'X2', 'Y2', 'ClassID']
                break

        # Read Images
        for image in os.listdir(path + folder):
            if not image.endswith('.csv'):
                tempdf2 = tempdf.loc[tempdf['Filename'] == str(image)]
                # Use these details to crop, resize and label
                img = cv2.imread(path + folder + '/' + image)
                crop_img = img[tempdf2.iloc[0]['Y1']:tempdf2.iloc[0]
                               ['Y2'], tempdf2.iloc[0]['X1']:tempdf2.iloc[0]['X2']]
#                 print(image, tempdf2.iloc[0]['Y1']-tempdf2.iloc[0]['Y2'], tempdf2.iloc[0]['X1']-tempdf2.iloc[0]['X2'], img.shape)
                resized = cv2.resize(crop_img, (28, 28))
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                images.append(gray)
                labels.append(tempdf2.iloc[0]['ClassID'])
        i += 1

    return images, labels

# In[7]:


print('Reading Train Images')
train_images, train_labels = read_images(train_folders, train_path)
print(' - Done')

print('Reading Test Images')
test_images, test_labels = read_images(test_folders, test_path)
print(' - Done')

print()
print('Train : Images - ', len(train_images), 'Labels - ', len(train_labels))
print('Test  : Images - ', len(test_images), 'Labels - ', len(test_labels))

# In[10]:

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

# In[11]:

# Create Mini Batches to feed the Network


def nextMiniBatch(images, labels, index, batch_size):
    imgs = []
    for img in images[index*batch_size: (index+1)*batch_size]:
        temp = np.reshape(img, (784))
        imgs.append(temp)

    labs = []
    for label in labels[index*batch_size: (index+1)*batch_size]:
        temp = np.zeros(62)
        temp[label] = 1
        labs.append(temp)

    return np.array(imgs), np.array(labs)

# Calculate Accuracy of Model after each Epoch


def calcAccuracy(images, labels):
    pred_y = tf.equal(tf.argmax(layer_out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_y, tf.float32))
    return accuracy.eval({x: images, y: labels})

# If either Cost or Accuracy is not Improving, Then Shut Down the model


def checkPatience(cost, accuracy, patience):
    if len(cost) < patience:
        return False
    for i in range(1, patience-1):
        if (cost[-i] != cost[-i-1]) or (accuracy[-i] != accuracy[-i-1]):
            return False
    return True

# In[12]:


def onehot(index, n_classes):
    label = np.zeros(n_classes)
    label[index] = 1
    return label


class DataGenerator(keras.utils.Sequence):
    def __init__(self, train_images, train_labels, batch_size=64, shuffle=False):
        self.batch_size = batch_size
        self.train_images = train_images
        self.train_labels = train_labels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.train_images) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        images = self.train_images[index *
                                   self.batch_size:(index+1)*self.batch_size]
        labels = self.train_labels[index *
                                   self.batch_size:(index+1)*self.batch_size]
        ilabels = []
        for l in labels:
            ilabels.append(onehot(l, 62))

        # Find list of IDs
#         images = [self.train_images[k] for k in indexes]
#         labels = [self.train_labels[k] for k in indexes]
        # Generate data
        return np.array(images), np.array(ilabels)

    def on_epoch_end(self):
        pass
#         self.indexes = np.arange(len(self.file_list))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)

# In[13]:

# Network Configuration


# Input Layer
n_input = 784
# Hidden Layers
n_hidden_1, n_hidden_2, n_hidden_3 = 256, 128, 256
# Output Layer
n_classes = len(train_folders)  # 62
# Parameters
learning_rate = 0.005

print('Network Configuration')
print('Input  - ', n_input)
print('Hidden - ', n_hidden_1, n_hidden_2, n_hidden_3)
print('Output - ', n_classes)
print('L Rate - ', learning_rate)

# In[28]:

# print('Creating Network', end='')
# Defining Network

input_c = keras.engine.input_layer.Input(shape=(28, 28))

layer1 = Flatten()(input_c)
layer2 = Dense(n_hidden_1, activation='sigmoid')(layer1)
layer3 = Dense(n_hidden_2, activation='sigmoid')(layer2)
layer4 = Dense(n_hidden_3, activation='sigmoid')(layer3)
output_c = Dense(n_classes, activation='softmax')(layer4)

model = Model(inputs=input_c, outputs=output_c)
adam = keras.optimizers.Adam(
    lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

# In[29]:

# Fit the model
train_generator = DataGenerator(train_images, train_labels, batch_size=128)
history = model.fit_generator(train_generator, epochs=20)

# In[30]:

# Save Complete Model
model.save('tensorflow-k.h5')

# In[31]:

# Predict Accuracy - Final Model
test_generator = DataGenerator(test_images, test_labels, batch_size=128)
accr = model.evaluate_generator(test_generator)

print('Loss: {:0.3f}\tAccuracy: {:0.3f}'.format(accr[0], accr[1]))

# Predict Accuracy - Best Model
# accr2 = loaded_model.evaluate(x=[test_imgs, test_msg], y=test_imgs, batch_size=32)
# print()
# print('Loss: {:0.3f}\tAccuracy: {:0.3f}'.format(accr2[0], accr2[1]))

# In[32]:

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.title('Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Accuracy'], loc='upper left')
plt.savefig('NN acc.png', dpi=300, pad_inches=0.1)
plt.show()

plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss'], loc='upper left')
plt.savefig('NN loss.png', dpi=300, pad_inches=0.1)
plt.show()

# In[33]:

# Show onw figure for each traffic sign
plt.figure(figsize=(25, 12))
plt.subplots_adjust(hspace=.1, wspace=.1)
for i in range(0, n_classes):
    index = np.where(train_labels == i)[0][0]
    image = train_images[index]
    plt.subplot(7, 10, i + 1), plt.imshow(image)
    plt.xticks([]), plt.yticks([])
plt.savefig('exploratory.jpg', dpi=300, pad_inches=0.1)
