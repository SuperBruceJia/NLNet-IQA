# -*- coding: utf-8 -*-

import random
import numpy as np
from numba import jit

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import to_tensor

from superpixel.slic import SLIC


def NonOverlappingCropPatches(im, args=None, transforms=True):
    """Non-overlapping Cropped Patches"""
    if args != None:
        patch_size = args.patch_size
    else:
        patch_size = 112

    w, h = im.size

    cnn_patches = ()
    gnn_patches = ()
    gnn_graphs = ()

    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            cropped_patch = im.crop((j, i, j + patch_size, i + patch_size))
            cnn_patch = to_tensor(cropped_patch)
            if transforms is True:
                cnn_patch = val_test_transforms(cnn_patch)
            cnn_patches += (cnn_patch,)

            SLIC_Class = SLIC(img=cropped_patch, args=args)
            superpixel_patch, patch_graph, _, _ = SLIC_Class.slic_function()

            gnn_patches += (superpixel_patch,)
            gnn_graphs += (patch_graph,)

    return torch.stack(cnn_patches), torch.stack(gnn_patches), torch.stack(gnn_graphs)


def RandomCropPatches(im, args=None, transforms=True):
    """Random Crop Patches"""
    if args != None:
        n_patches_train = args.n_patches_train
        patch_size = args.patch_size
    else:
        n_patches_train = 1
        patch_size = 112

    w, h = im.size

    cnn_patches = ()
    gnn_patches = ()
    gnn_graphs = ()
    for i in range(n_patches_train):
        w1 = np.random.randint(low=0, high=w - patch_size + 1)
        h1 = np.random.randint(low=0, high=h - patch_size + 1)

        cropped_patch = im.crop((w1, h1, w1 + patch_size, h1 + patch_size))
        cnn_patch = to_tensor(cropped_patch)
        if transforms is True:
            cnn_patch = train_transforms(cnn_patch)
        cnn_patches += (cnn_patch,)

        SLIC_Class = SLIC(img=cropped_patch, args=args)
        superpixel_patch, patch_graph, _, _ = SLIC_Class.slic_function()

        gnn_patches += (superpixel_patch, )
        gnn_graphs += (patch_graph, )

    return torch.stack(cnn_patches), torch.stack(gnn_patches), torch.stack(gnn_graphs)


def CropPatchesTesting(im, args=None, transforms=True):
    """Non-overlapping Cropped Patches"""
    if args != None:
        patch_size = args.patch_size
    else:
        patch_size = 112

    w, h = im.size

    cnn_patches = ()
    gnn_patches = ()
    gnn_graphs = ()
    image_centers = []
    img_slics = []

    for i in range(0, h - patch_size, patch_size):
        for j in range(0, w - patch_size, patch_size):
            cropped_patch = im.crop((j, i, j + patch_size, i + patch_size))
            cnn_patch = to_tensor(cropped_patch)
            if transforms is True:
                cnn_patch = val_test_transforms(cnn_patch)
            cnn_patches = cnn_patches + (cnn_patch,)

            SLIC_Class = SLIC(img=cropped_patch, args=args)
            superpixel_patch, patch_graph, image_centers_loc, img_slic = SLIC_Class.slic_function()

            gnn_patches = gnn_patches + (superpixel_patch,)
            gnn_graphs = gnn_graphs + (patch_graph,)
            image_centers.append(image_centers_loc)
            img_slics.append(img_slic)

    return torch.stack(cnn_patches), torch.stack(gnn_patches), torch.stack(gnn_graphs), image_centers, img_slics


train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


val_test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
