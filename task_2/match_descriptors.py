# reference: https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590
from matplotlib import pyplot as plt
import numpy as np
import cv2

# Uncomment and comment the images you want to use

#img1= cv2.imread('HG_no_grid_04.jpg', 0)
#img2 = cv2.imread('HG_no_grid_05.jpg', 0)

img1= cv2.imread('HG_02.jpg', 0)
img2 = cv2.imread('HG_03.jpg', 0)

# Uses ORB descriptors to compute the feature matches in each image
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# matcher takes normType, cv2.NORM_HAMMING for ORB, FAST and BRIEF
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # brute-force matcher function
matches = bf.match(des1, des2) # matches the points between images
matches = sorted(matches, key=lambda x: x.distance)

# draw first 50 matches
match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, matchColor=(0,255,0), flags=2)

# Save Images (uncomment corresponding to image used)
#cv2.imwrite('matches_no_grid.jpg', match_img)
cv2.imwrite('matches_grid.jpg', match_img)

# Show the images
cv2.imshow('Keypoint correspondences of images without grid', match_img)
cv2.waitKey()
