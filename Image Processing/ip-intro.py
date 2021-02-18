#!/usr/bin/env python
# coding: utf-8

# Image Processing / ip-intro.py
# Gourav Siddhad
# 15-Feb-2019

# In[1]:
# Introduction to Image Processing

import skimage as si
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
from skimage import transform as tf

# In[2]:
# Image................

i1 = np.zeros([256,256])

io.imshow(i1)
plt.show()

for i in range(5,250):
    for j in range(5,250):
        i1[i,j] = 1;

io.imshow(i1)
plt.show()

# In[3]:
# An example -----

img1 = io.imread('img02.jpg')
io.imshow(img1)
print("Image Shape:",img1.shape)
plt.show()

for i in range(195,350):
    for j in range(195,290):
        img1[i,j,1]=0
        img1[i,j,0]=0
        
io.imshow(img1)

plt.show()

# In[6]:
# ------------------Smoothening------------------

mask = [[0,1,0],
        [1,4,1],
        [0,1,0]]

mask1 = np.array(mask)
mask1

img3 = io.imread('sharp.png')
io.imshow(img3)
print(img3.shape)
plt.show()

x,y=300,300
img3 = si.color.rgb2gray(img3)
p = tf.resize(img3,(x,y))
print("Image Shape:",p.shape)
io.imshow(p)
plt.show()

p_mean = np.empty((x,y))

for i in range(x-1): # why x-1 and y-1
    for j in range(y-1):
        p_mean[i,j]= np.mean([p[i-1,j-1],p[i-1,j],p[i-1,j+1],p[i,j-1],p[i,j],p[i,j+1],p[i+1,j-1],p[i+1,j],p[i+1,j+1]])

io.imshow(p_mean)
plt.show()

# In[8]:

mbd3 = io.imread('e1.png')
io.imshow(mbd3)
plt.show()
mbd3.shape

# In[9]:

from skimage.filters.rank import median
from skimage.morphology import disk

output = median(mbd3,disk(1)) # print disk............and check

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(mbd3,cmap=plt.cm.gray)
ax[1].imshow(output,cmap=plt.cm.gray)    

plt.tight_layout()
plt.show()

# In[10]:
# -----------Image sharpening----------

brain = io.imread('blur.jpg')
io.imshow(brain)
plt.show()
brain = si.color.rgb2gray(brain )
brain.shape

p_laplac = si.filters.laplace(brain,3)
p_added_laplac = p_laplac + brain
io.imshow(abs(p_laplac),cmap=plt.cm.gray)
plt.show()

# In[12]:

io.imshow(abs(p_added_laplac),cmap=plt.cm.gray)
plt.show()

# In[13]:
# -----------------Dilation--------------

from skimage.morphology import binary_dilation

img4 = io.imread('panda2.png')
img4 = si.color.rgb2gray(img4)
io.imshow(img4)

plt.show()

morph = [[0,0,0],[0,0,0],[0,0,0]]
dilated = binary_dilation(img4,disk(2)) #use morph also--------------------

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(img4,cmap=plt.cm.gray)
ax[1].imshow(dilated,cmap=plt.cm.gray)    

plt.tight_layout()
plt.show()

# In[14]:
# -----------------Erosion ----------------

from skimage.morphology import erosion

img5 = io.imread('panda2.png')
img5 = si.color.rgb2gray(img5)
io.imshow(img5)

plt.show()

morph = [[1,1,1],[1,1,1],[1,1,1]]
eroded = erosion(img5, morph) # use disk also---------------

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(img5,cmap=plt.cm.gray)
ax[1].imshow(eroded ,cmap=plt.cm.gray)    

plt.tight_layout()
plt.show()

# In[16]:

x,y = img5.shape
new = np.zeros([x,y])
for i in range(x):
    for j in range(y):
        if img5[i,j] == 0:
            new[i,j] = 1        

# In[17]:

io.imshow(new)
plt.show()

# In[18]:
morph = [[1,1,1],[1,1,1],[1,1,1]]
eroded = erosion(new, morph) # use disk also---------------

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(new,cmap=plt.cm.gray)
ax[1].imshow(eroded ,cmap=plt.cm.gray)    

plt.tight_layout()
plt.show()

# In[19]:
# ------------- Erosion by reversal--------------

new2 = np.zeros([x,y])
for i in range(x):
    for j in range(y):
        if new[i,j] == 0:
            new2[i,j] = 1 

# In[20]:

io.imshow(new2)
plt.show()

# In[21]:

from skimage.morphology import erosion

img5 = io.imread('panda2.png')
img5 = si.color.rgb2gray(img5)
io.imshow(img5)

plt.show()

morph = [[1,1,1],[1,1,1],[1,1,1]]
eroded = erosion(img5, morph) # use disk also---------------

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(img5,cmap=plt.cm.gray)
ax[1].imshow(eroded ,cmap=plt.cm.gray)    

plt.tight_layout()
plt.show()

# In[22]:
# -------------Opening--------------------

from skimage.morphology import opening

img6 = io.imread('panda2.png')
img6 = si.color.rgb2gray(img6)
io.imshow(img6)

plt.show()

morph = [[1,1,1],[1,1,1],[1,1,1]]
opened = opening(img6, disk(2)) # use disk also---------------

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(img6,cmap=plt.cm.gray)
ax[1].imshow(opened ,cmap=plt.cm.gray)    

plt.tight_layout()
plt.show()

# In[23]:
# # ------------Closing--------------------

from skimage.morphology import closing

img7 = io.imread('panda2.png')
img7 = si.color.rgb2gray(img7)
io.imshow(img7)

plt.show()

morph = [[1,1,1],[1,1,1],[1,1,1]]
closed = closing(img7, disk(1)) # use disk also---------------

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(img7,cmap=plt.cm.gray)
ax[1].imshow(closed ,cmap=plt.cm.gray)    

plt.tight_layout()
plt.show()
