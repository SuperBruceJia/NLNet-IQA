# -*- coding: utf-8 -*-

import cv2
import random
import numpy as np
from skimage.measure import regionprops
from numba import jit
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F

from lib.utils import local_normalize


class SLIC:
    """
    SLIC Superpixel Segmentation Algorithm
    """
    def __init__(self, img, args):
        self.img = np.array(img)
        self.normalize_img = local_normalize(img=self.img, num_ch=3, const=127.0)

        self.image_n_nodes = args.image_n_nodes
        self.patch_n_nodes = args.patch_n_nodes

        self.region_size = args.region_size
        self.n_features = args.n_features

        self.ruler = args.ruler
        self.iterate = args.iterate

    def slic_function(self):
        """
        Main Function
        """
        self.img_lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        slic = cv2.ximgproc.createSuperpixelSLIC(image=self.img_lab, region_size=self.region_size, ruler=self.ruler)
        slic.iterate(self.iterate)

        # Get the labels of superpixels and number of patches
        self.label = slic.getLabels()
        mask_slic = slic.getLabelContourMask(thick_line=True)
        mask_inv_slic = cv2.bitwise_not(mask_slic)
        img_slic = cv2.bitwise_and(self.img, self.img, mask=mask_inv_slic)

        # Remove the index that has fewer nodes
        self.num_clusters = slic.getNumberOfSuperpixels()
        all_cluster = list(range(0, self.num_clusters))
        cleaned_clusters = self.remove_cluster(clusters=all_cluster)

        # Get image-level and patch-level info
        patch, centers_loc = self.get_patch(clusters=cleaned_clusters)

        # Build patch-related graph -> complete graph
        # Note: In our work, we only used complete graph to build Graph
        patch_graph = torch.ones([self.image_n_nodes, self.patch_n_nodes, self.patch_n_nodes])
        # patch_graph = build_patch_graph(centers=patches_loc,
        #                                 image_n_nodes=self.image_n_nodes,
        #                                 patch_n_nodes=self.patch_n_nodes,
        #                                 region_size=self.region_size)

        return torch.Tensor(patch), torch.Tensor(patch_graph), centers_loc, img_slic

    def get_patch(self, clusters):
        """
        Get Superpixel Patches and their center locations
        """
        centers_loc = []
        patches = []
        image_patch_index = []

        # Get image patches
        cluster_index = np.linspace(0, len(clusters) - 1, self.image_n_nodes, dtype=int)
        for cluster_id in cluster_index:
            image_patch_index.append(clusters[cluster_id])

        for props in image_patch_index:
            pixel_loc = np.where(self.label == props)
            superpixels_patch = []

            # Get superpixels' patch with `patch_nodes` nodes
            index = np.linspace(0, np.shape(pixel_loc)[1] - 1, self.patch_n_nodes, dtype=int)
            for i in index:
                superpixels_patch.append(self.normalize_img[pixel_loc[0][i]][pixel_loc[1][i]])

            patches.append(superpixels_patch)

            # Get patches' center location and pixels
            cx = int(np.mean(pixel_loc[0]))
            cy = int(np.mean(pixel_loc[1]))
            centers_loc.append([cx, cy])

        return patches, centers_loc

    def remove_cluster(self, clusters):
        """
        Remove the clusters with fewer nodes (than num_nodes)
        """
        temp = clusters.copy()
        for id in clusters:
            loc = np.where(self.label == id)
            if np.shape(loc)[1] <= self.patch_n_nodes:
                temp.remove(id)

        return temp


# Note: I only wrote the following functions to implement ideas.
# In this work, these functions are not used
# and I simply used the complete graph to model the Graph inside a superpixel patch
# and Cosine Similarity to model the Graph among Superpixel patches.
@jit
def build_image_graph(centers, image_n_nodes, region_size):
    """
    Build Graph among superpixel patches
    """
    centers = torch.as_tensor(centers)

    graph = np.zeros([image_n_nodes, image_n_nodes])
    for node in range(image_n_nodes):
        l2_dis = np.array(F.pairwise_distance(x1=centers[node], x2=centers, p=2))
        graph[node, :][np.where(l2_dis <= region_size)[0]] = 1

    return graph


# @jit
def build_patch_graph(centers, image_n_nodes, patch_n_nodes, region_size):
    """
    Build Graph inside a superpixel patch
    """
    # graphs = []
    graphs = ()
    # centers = torch.Tensor(centers)
    threshold = region_size * region_size / patch_n_nodes

    for patch_id in range(image_n_nodes):
        # graph = np.zeros([patch_n_nodes, patch_n_nodes])
        graph = torch.zeros([patch_n_nodes, patch_n_nodes])

        # Iterate all the nodes (num_nodes)
        for node in range(patch_n_nodes):
            # l2_dis = np.array(F.pairwise_distance(x1=centers[patch_id][node], x2=centers[patch_id], p=2))
            # graph[node, :][np.where(l2_dis <= threshold)[0]] = 1

            l2_dis = F.pairwise_distance(x1=centers[patch_id][node], x2=centers[patch_id], p=2)
            graph[node, :] = torch.where(l2_dis <= threshold, 1.0, 0.0)
            # graph[node, :][np.where(l2_dis <= threshold)[0]] = 1

        # graphs.append(graph)
        graphs += (graph,)

    # return graphs
    return torch.stack(graphs)