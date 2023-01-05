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


img = cv2.imread("test_images/cr7.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# SLIC Superpixel Segmentation
slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=55, ruler=40.0)
slic.iterate(10)

mask_slic = slic.getLabelContourMask()
label_slic = slic.getLabels()
number_slic = slic.getNumberOfSuperpixels()
mask_inv_slic = cv2.bitwise_not(mask_slic)
img_slic = cv2.bitwise_and(img, img, mask=mask_inv_slic)

plt.imshow(img_slic)
plt.show()

# SEEDS Superpixel Segmentation
seeds = cv2.ximgproc.createSuperpixelSEEDS(image_width=img.shape[1],
                                          image_height=img.shape[0],
                                          image_channels=img.shape[2],
                                          num_superpixels=25,
                                          num_levels=20,
                                          prior=5,
                                          histogram_bins=20,
                                          double_step=True)
seeds.iterate(img, 10)
mask_seeds = seeds.getLabelContourMask()


label_seeds = seeds.getLabels()
label_seeds[(label_seeds == 0.)] = label_seeds.max() + 1
regions = regionprops(label_slic)


centroids = []
for props in regions:
    cx, cy = props.centroid  # centroid coordinates
    centroids.append([cx, cy])
print('The number of patches: ', len(centroids))

number_seeds = seeds.getNumberOfSuperpixels()
mask_inv_seeds = cv2.bitwise_not(mask_seeds)
img_seeds = cv2.bitwise_and(img, img, mask=mask_inv_seeds)

# Scatter centroid of each superpixel
plt.imshow(img_seeds)
plt.scatter([x.centroid[1] for x in regions], [y.centroid[0] for y in regions], c='red')
plt.show()

