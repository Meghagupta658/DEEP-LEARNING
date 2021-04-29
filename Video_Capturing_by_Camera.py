#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 18:10:56 2021

@author: meghagupta
"""

#capturing videos, pictures and images from camera using openCV
import cv2
import matplotlib.pyplot as plt

capture = cv2.VideoCapture(0)  #VideoCapture function in cv2 lib for capturing video by camera
ret, img = capture.read()   #this will capture image by camera
plt.imshow(img)

#Capturing Video by Camera:
capture = cv2.VideoCapture(0) 
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('width',width,'height',height)
while True:
    ret, frame = capture.read() 
    grayscale =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',grayscale)
    if cv2.waitKey(1) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()




    



