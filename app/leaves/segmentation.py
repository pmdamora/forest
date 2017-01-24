# Forest
# Copyright 2016 pauldamora.me All rights reserved
#
# Authors: Paul D'Amora
#
# Description: Detects leaf boundaries and removes everything not in boundary
# Sets everything inside boundary to black, and removes noise
import os
import numpy as np
from skimage import io
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import cv2

# Define constants
IMAGE_WIDTH = 300.0

# Load image
filename = os.path.join('./input', 'leaf.jpg')
image = cv2.imread(filename)

# Compute the ratio of the old image width to the new image width (300)
# Make a copy of the original image, and resize image to 300
ratio = IMAGE_WIDTH / image.shape[1]
dim = (int(IMAGE_WIDTH), int(image.shape[0] * ratio))
orig = image.copy()
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# Convert the image to grayscale, blur it, and find edges
# in the image using canny edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

# Find contours in the image, only keep the largest ones
(_, contours, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
leaf_contour = None

contour_image = cv2.drawContours(image.copy(), [leaf_contour], -1,
                                 (0, 255, 0), 3)
cv2.imshow("Resize", image)
cv2.imshow("Gray", gray)
cv2.imshow("Edge", edged)
cv2.imshow("Contours", contour_image)
cv2.waitKey(0)
