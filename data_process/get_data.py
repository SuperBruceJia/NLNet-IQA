# -*- coding: utf-8 -*-

import time
import os
import h5py
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from torchvision import models, transforms

from superpixel.slic import SLIC
from lib.make_index import make_index, default_loader
from lib.image_process import RandomCropPatches, NonOverlappingCropPatches
from lib.utils import mos_rescale


class IQADataset(Dataset):
    """
    IQA Dataset
    """
    def __init__(self, args, status='train', loader=default_loader):
        """
        :param args: arguments of the model
        :param status: train/val/test
        :param loader: image loader
        """
        self.status = status
        self.loader = loader

        self.args = args
        self.database = args.database

        self.image_n_nodes = args.image_n_nodes
        self.patch_n_nodes = args.patch_n_nodes
        self.region_size = args.region_size
        self.ruler = args.ruler
        self.iterate = args.iterate

        self.patch_size = args.patch_size
        self.n_patches_train = args.n_patches_train

        Info = h5py.File(args.data_info, 'r')
        index = Info['index']
        index = index[:, 0 % index.shape[1]]
        ref_ids = Info['ref_ids'][0, :]

        # Get dataset index
        trainindex, valindex, testindex = make_index(args=args, index=index)

        if 'train' in status:
            print('*' * 100)
            print('The training set indexes are ', trainindex)
            print('The validation set indexes are ', valindex)
            print('The testing set indexes are ', testindex)
            print('*' * 100, '\n')

        # Split Database and make sure there are no contents overlap
        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            if ref_ids[i] in testindex:
                test_index.append(i)

            elif ref_ids[i] in valindex:
                val_index.append(i)

            else:
                train_index.append(i)

        if 'train' in status:
            self.index = train_index
            print('*' * 100)
            print("Number of Training Images: {}".format(len(self.index)))
            print('The Training set indexes are ', self.index, '\n\n')

        if 'val' in status:
            self.index = val_index
            print("Number of Validation Images: {}".format(len(self.index)))
            print('The Validation set indexes are ', self.index, '\n\n')

        if 'test' in status:
            self.index = test_index
            print("Number of Testing Images: {}".format(len(self.index)))
            print('The Testing set indexes are ', self.index, '\n\n')
            print('*' * 100, '\n')

        self.mos = Info['subjective_scores'][0, self.index]
        self.mos_std = Info['subjective_scoresSTD'][0, self.index]
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]

        self.patches = ()
        self.label = []
        self.label_std = []
        self.im_names = []
        self.dis_type = []

        # Get image names and their scores
        for idx in range(len(self.index)):
            self.im_names.append(os.path.join(args.im_dir, im_names[idx]))
            self.label.append(self.mos[idx])
            self.label_std.append(self.mos_std[idx])

            if self.database == 'TID2008' or self.database == 'TID2013':
                self.dis_type.append(int(im_names[idx][4:6]) - 1)

            elif self.database == 'KADID':
                self.dis_type.append(int(im_names[idx][4:6]) - 1)

            elif self.database == 'CSIQ':
                # Distortion Type
                if 'AWGN' in im_names[idx]:
                    self.dis_type.append(0)
                elif 'BLUR' in im_names[idx]:
                    self.dis_type.append(1)
                elif 'contrast' in im_names[idx]:
                    self.dis_type.append(2)
                elif 'fnoise' in im_names[idx]:
                    self.dis_type.append(3)
                elif 'JPEG' in im_names[idx]:
                    self.dis_type.append(4)
                elif 'jpeg2000' in im_names[idx]:
                    self.dis_type.append(5)

            elif self.database == 'LIVE':
                # Distortion Type
                if 'jp2k' in im_names[idx]:
                    self.dis_type.append(0)
                elif 'jpeg' in im_names[idx]:
                    self.dis_type.append(1)
                elif 'wn' in im_names[idx]:
                    self.dis_type.append(2)
                elif 'gblur' in im_names[idx]:
                    self.dis_type.append(3)
                elif 'fastfading' in im_names[idx]:
                    self.dis_type.append(4)

            elif self.database == 'SIQAD':
                loc = im_names[idx].find('_')
                self.dis_type.append(int(im_names[idx][loc + 1]) - 1)

            elif self.database == 'SCID':
                self.dis_type.append(int(im_names[idx][6]) - 1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        im = self.loader(self.im_names[idx])

        # start = time.time()
        # Get CNN input
        if self.status == 'train':
            cnn_input, patch, patch_graph = RandomCropPatches(im, args=self.args, transforms=True)
        else:
            cnn_input, patch, patch_graph = NonOverlappingCropPatches(im, args=self.args, transforms=True)

        # Get labels
        label = torch.as_tensor([self.label[idx], ])
        label_std = torch.as_tensor([self.label_std[idx], ])

        # Choose whether to use distortion type or distortion level
        dis_type = torch.as_tensor([self.dis_type[idx], ])

        return patch, patch_graph, cnn_input, label, label_std, dis_type
