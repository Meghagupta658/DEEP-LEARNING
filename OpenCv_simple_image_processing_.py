#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 13:35:33 2021

@author: meghagupta
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image      #PIL is Python Image Library
pic = Image.open('/Users/meghagupta/Downloads/Pic1.jpeg')
pic

pic_arr = np.asarray(pic)   # to convert jpeg to array type
pic_arr

plt.imshow(pic_arr)     #imshow is a fuhnction in matplot lib for ploting images
pic_rgb = pic_arr.copy()

#we have 3 color in computer vision that red, green and blue. Red=0 , Greeen=1 and Blue =2
plt.imshow(pic_rgb[:,:,0])  #by this red color is removed from the image 
plt.imshow(pic_rgb[:,:,0],cmap='gray') #it will just give gray image no color

plt.imshow(pic_rgb[:,:,1])   # it will give more of green
plt.imshow(pic_rgb[:,:,2])  # it will give more of blue

pic_rgb[:,:,1]=0
plt.imshow(pic_rgb)   #there is no green color, only red and blue color.So looks purple

pic_rgb[:,:,2]=0
plt.imshow(pic_rgb)   #there is no blue color, only red and green color


#cv2 is an openCV lib
import cv2
img = cv2.imread('/Users/meghagupta/Downloads/Pic1.jpeg')
plt.imshow(img)   
       
#matplotlib----- read R G B
#opencv----- read B G R
#So we need to tranform

fix_img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img) 

plt.imshow(cv2.resize(fix_img,(1000,1000)))   # for size change : resize
plt.imshow(cv2.flip(fix_img, 0))
fix_img.shape

fix_img[60:500 ,500:550] = (60,10,88)   #by this we can create different color on image
plt.imshow(fix_img)

fix_img[60:500, 80:100] = (200,10,88)   #by this we can create different color image on different position  # left side is controlling postion of color image and right side gives the color to image
plt.imshow(fix_img)

#for creating different images
canvas = np.zeros((300,300,3),dtype='uint8')
plt.imshow(canvas)

#creating images on canvas
cv2.line(canvas, (0,0), (300,300), (0,255,0), 5)    #creating line
cv2.imshow('the canvas', canvas)
cv2.waitKey(0)

cv2.rectangle(canvas, (10,10), (70,70), (0,255,0), 5)      #creating rectangle
cv2.imshow('the canvas', canvas)
cv2.waitKey(0)

cv2.circle(canvas, (150,150), 50, (0,255,0), 5)    #creating circle
cv2.imshow('the canvas', canvas)
cv2.waitKey(0)
