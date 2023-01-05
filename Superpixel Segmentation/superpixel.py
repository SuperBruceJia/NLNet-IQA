#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops


def image_show(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# Read the image
img = cv2.imread("test_images/cr7.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# SLIC Superpixel Segmentation
slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=32, ruler=40.0)
slic.iterate(10)

mask_slic = slic.getLabelContourMask()
label_slic = slic.getLabels()
label_slic[(label_slic == 0.)] = label_slic.max() + 1
regions = regionprops(label_slic)

# Get superpixels' centroids
centroids = []
for props in regions:
    cx, cy = props.centroid  # centroid coordinates
    centroids.append([cx, cy])
print('The number of patches: ', len(centroids))

number_slic = slic.getNumberOfSuperpixels()
mask_inv_slic = cv2.bitwise_not(mask_slic)
img_slic = cv2.bitwise_and(img, img, mask=mask_inv_slic)
plt.imshow(img_slic)
plt.show()
