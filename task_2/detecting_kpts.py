# Method inspired from https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590
from matplotlib import pyplot as plt
import numpy as np
import cv2

## Load image

#HG (rotation) with grid 
img = cv2.imread('HG_05.jpg')

#HG without grid 
#img = cv2.imread('HG_no_grid_05.jpg') # uncomment to use image without the grid

## This is the Haris Corner Detection Method
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # makes the image into gray-scale
dst = cv2.cornerHarris(gray_img, blockSize=2, ksize=(3), k=0.04)

# dilate to mark the corners
dst = cv2.dilate(dst, None)
img[dst > 0.01 * dst.max()] = [0, 255, 0]

## This is the ORB method
orb = cv2.ORB_create(nfeatures=500) # detects 1000 feature points in the image
kp, des = orb.detectAndCompute(gray_img, None)
kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0) # draws the keypoinys as green points

## Saving resulting images 

#For image with grid
cv2.imwrite('haris__grid.jpg', img)
cv2.imwrite('ORB_grid.jpg', kp_img)

#For image without grid (uncomment)
#cv2.imwrite('haris_no_grid.jpg', kp_img)
#cv2.imwrite('ORB_no_grid.jpg', img)

# Show the Images
cv2.imshow('Haris Corner Detection Method', img)
cv2.imshow('ORB Method', kp_img)
cv2.waitKey(0)
