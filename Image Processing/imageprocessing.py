#!/usr/bin/env python
# coding: utf-8

# Image Processing / imageprocessing.py
# Gourav Siddhad
# 15-Feb-2019

# 01.1 Draw 'F' as an image.(do not load an image of F)
# 01.2 Draw a bank chess board
# 01.3 take any rgb image. and show in 9 blocks as follows:
#      [[Red block][Green Block][Blue Block]]
#      [[Green block][Blue Block][Red Block]]
#      [[Blue block][Red Block][Green Block]]

# 02. I have one letter as an image(img02.jpg) form. I am not able to read it.
#     provide more clear picture of letter, using appropriate operations.

# 03. Image (img03.jpg) contains too many dots, which makes the picture very noisey. Clean the image without loosing coins.

# 04. Update the side face detection code to detect left-facing also.

# In[ ]:

import skimage as si
from skimage import io
from skimage import transform as tf

from matplotlib import pyplot as plt
import numpy as np

# In[ ]:

# 01.1 Draw 'F' as an image.(do not load an image of F)

imgF = np.zeros([256,256])

for j in range(80,205):
    for i in range(5,10):
        imgF[i,j] = 1
        
for j in range(80,190):        
    for i in range(128,133):
        imgF[i,j] = 1
        
for i in range(5,250):
    for j in range(80,85):
        imgF[i,j] = 1
    
io.imshow(imgF)
plt.show()

# In[ ]:

# 01.2 Draw a bank chess board

cBoard = np.zeros([256,256])

lenCB = int(256/8)
for i in range(0,8):
    for j in range(0,8):
        if (i+j) % 2 is 0:
            for k in range(0,lenCB):
                for l in range(0, lenCB):
                    cBoard[lenCB*i+k, lenCB*j+l] = 1
    
io.imshow(cBoard)
plt.show()

# In[ ]:

# 01.3 take any rgb image. and show in 9 blocks as follows:
##      [[Red block][Green Block][Blue Block]]
##      [[Green block][Blue Block][Red Block]]
##      [[Blue block][Red Block][Green Block]]

srcimg = io.imread("sample.jpg")
outimg = srcimg.copy()
outlen, outwid = outimg.shape[:2]

lsize = int(outlen/3)
wsize = int(outwid/3)

# Red Block   1,1
outimg[0:lsize,0:wsize,1] = 0
outimg[0:lsize,0:wsize,2] = 0

# Green Block 1,2
outimg[0:lsize,wsize:2*wsize,0] = 0
outimg[0:lsize,wsize:2*wsize,2] = 0

# Blue Block  1,3
outimg[0:lsize,2*wsize:3*wsize,0] = 0
outimg[0:lsize,2*wsize:3*wsize,1] = 0

# Green block 2,1
outimg[lsize:2*lsize,0:wsize,0] = 0
outimg[lsize:2*lsize,0:wsize,2] = 0

# Blue Block  2,2
outimg[lsize:2*lsize,wsize:2*wsize,0] = 0
outimg[lsize:2*lsize,wsize:2*wsize,1] = 0

# Red Block   2,3
outimg[lsize:2*lsize,2*wsize:3*wsize,1] = 0
outimg[lsize:2*lsize,2*wsize:3*wsize,2] = 0

# Blue block  3,1
outimg[2*lsize:3*lsize,0:wsize,0] = 0
outimg[2*lsize:3*lsize,0:wsize,1] = 0

# Red Block   3,2
outimg[2*lsize:3*lsize,wsize:2*wsize,1] = 0
outimg[2*lsize:3*lsize,wsize:2*wsize,2] = 0

# Green Block 3,3
outimg[2*lsize:3*lsize,2*wsize:3*wsize,0] = 0
outimg[2*lsize:3*lsize,2*wsize:3*wsize,2] = 0

print('Original image')
io.imshow(srcimg)
plt.show()

print('Output image')
io.imshow(outimg)
plt.show()

# In[ ]:

# 02. I have one letter as an image(img02.jpg) form. 
# I am not able to read it. provide more clear picture of letter, using appropriate operations.

from skimage.morphology import erosion, dilation, disk

imgLetter = io.imread('img02.jpg')
imgLetter = si.color.rgb2gray(imgLetter)

output = erosion(imgLetter,disk(1))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 20),sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(imgLetter,cmap=plt.cm.gray)
ax[1].imshow(output,cmap=plt.cm.gray)    

plt.tight_layout()
plt.show()

# In[ ]:

# 03. Image (img03.jpg) contains too many dots, which makes the picture very noisey. Clean the image without loosing coins.

from skimage.morphology import opening, closing

imgCoin = io.imread('img03.jpg')
imgCoin = si.color.rgb2gray(imgCoin)

closed = closing(imgCoin,disk(2))
opened = opening(closed, disk(2))

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5),sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(imgCoin,cmap=plt.cm.gray)
ax[1].imshow(opened,cmap=plt.cm.gray)    

plt.tight_layout()
plt.show()

# In[ ]:

# 04. Update the side face detection code to detect left-facing also.

import cv2
import sys

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # Front face
pface_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")        # Profile face

video_capture = cv2.VideoCapture(0)
vwidth = int(video_capture.get(3))
vheight = int(video_capture.get(4))
print("Frame Size : ", vwidth, vheight)

while True:
    # Capture frame-by-frame
    ret, image = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (30,30)) # Front Face
    pfaces1 = pface_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (30,30)) # Left face
    flipgray = cv2.flip(gray,1)
    pfaces2 = pface_cascade.detectMultiScale(flipgray, scaleFactor = 1.2, minNeighbors = 5, minSize = (30,30)) # Right face
 
    for (x,y,w,h) in faces:   # Front face
        cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2) # Blue
    for (x,y,w,h) in pfaces1:   # Left Side face
        cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2) # Green
    for (x,y,w,h) in pfaces2:   # Right Side face
        cv2.rectangle(image, (vwidth-x, y), (vwidth-x-w, y+h), (0, 0, 255), 2) # Red

    cv2.imshow("Faces found", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
