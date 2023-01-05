#!/usr/bin/env python
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
from torchvision import models, transforms

from superpixel.slic import SLIC
from lib.make_index import make_data_index, default_loader
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
        self.args = args
        self.status = status
        self.loader = loader

        self.train_database = args.train_database
        self.test_database = args.test_database

        self.image_n_nodes = args.image_n_nodes
        self.patch_n_nodes = args.patch_n_nodes
        self.region_size = args.region_size
        self.ruler = args.ruler
        self.iterate = args.iterate

        self.patch_size = args.patch_size
        self.n_patches_train = args.n_patches_train

        # Train
        train_Info = h5py.File(args.train_info, 'r')
        train_index = train_Info['index']
        train_index = train_index[:, 0 % train_index.shape[1]]
        train_ref_ids = train_Info['ref_ids'][0, :]

        # Test
        test_Info = h5py.File(args.test_info, 'r')
        test_index = test_Info['index']
        test_index = test_index[:, 0 % test_index.shape[1]]
        test_ref_ids = test_Info['ref_ids'][0, :]

        # Get dataset index
        train_index_, test_index_ = [], []
        if 'train' in status:
            print('The Training Set Index is ', train_index, ' and num of Training index is ', len(train_index))
            for i in range(len(train_ref_ids)):
                train_index_.append(i)

            self.index = train_index_
            print("Number of Training Images: {}\n".format(len(self.index)))
            self.mos = train_Info['subjective_scores'][0, self.index]
            # self.mos_std = train_Info['subjective_scoresSTD'][0, self.index]
            im_names = [train_Info[train_Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]

            self.label = []
            self.im_names = []
            self.dis_type = []

            for idx in range(len(self.index)):
                self.im_names.append(os.path.join(args.train_im_dir, im_names[idx]))
                self.label.append(self.mos[idx])

                if self.train_database == 'TID2008' or self.train_database == 'TID2013':
                    self.dis_type.append(int(im_names[idx][4:6]) - 1)

                elif self.train_database == 'KADID':
                    self.dis_type.append(int(im_names[idx][4:6]) - 1)

                elif self.train_database == 'CSIQ':
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

                elif self.train_database == 'LIVE':
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

                elif self.train_database == 'SIQAD':
                    loc = im_names[idx].find('_')
                    self.dis_type.append(int(im_names[idx][loc + 1]) - 1)

                elif self.train_database == 'SCID':
                    self.dis_type.append(int(im_names[idx][6]) - 1)

        else:
            print('The Testing Set Index is ', test_index, ' and num of test index is ', len(test_index))
            for i in range(len(test_ref_ids)):
                test_index_.append(i)

            self.index = test_index_
            print("Number of Testing Images: {}".format(len(self.index)), '\n')
            self.mos = test_Info['subjective_scores'][0, self.index]
            im_names = [test_Info[test_Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]

            self.label = []
            self.im_names = []
            self.dis_type = []

            for idx in range(len(self.index)):
                self.im_names.append(os.path.join(args.test_im_dir, im_names[idx]))
                self.label.append(self.mos[idx])
                self.dis_type.append(0)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        im = self.loader(self.im_names[idx])

        # Get CNN input
        if self.status == 'train':
            cnn_input, patch, patch_graph = RandomCropPatches(im, args=self.args, transforms=True)
        else:
            cnn_input, patch, patch_graph = NonOverlappingCropPatches(im, args=self.args, transforms=True)

        # Get labels
        label = torch.as_tensor([self.label[idx], ])

        # Choose whether to use distortion type or distortion level
        dis_type = torch.as_tensor([self.dis_type[idx], ])

        return patch, patch_graph, cnn_input, label, dis_type
