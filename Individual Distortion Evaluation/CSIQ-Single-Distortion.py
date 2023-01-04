#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio
import mat73

import time
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from lib.utils import *
from benchmark.database import database
from model.solver import GNNSolver
from model.network import Network
from superpixel.fast import SLIC
from lib.image_process import NonOverlappingCropPatches
from lib.make_index import default_loader


def run(args):
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    start = time.time()

    # CSIQ
    CSIQinfo = mat73.loadmat('benchmark/CSIQfullinfo.mat')
    label_info = CSIQinfo["subjective_scores"]
    images_info = CSIQinfo["im_names"]

    distortion_types = ['contrast', 'awgn', 'blur', 'fnoise', 'jpeg', 'jpeg2000']
    for individual_type in distortion_types:
        srcc_all = []
        plcc_all = []
        for model_id in range(10):
            # Load model
            args.model_file = './save_model/CSIQ-32-4-' + str(model_id + 1) + '.pth'
            model = Network(args=args).cuda()
            model.load_state_dict(torch.load(args.model_file))
            model.train(False)

            pred_scores = []
            gt_scores = []
            file_dir = '/media/shuyuej/Projects/Dataset/CSIQ/' + individual_type + '/'
            for img_id in os.listdir(file_dir):
                img_path = file_dir + img_id
                im = default_loader(path=img_path)

                img_name = [individual_type + '/' + img_id]
                if img_name in images_info:
                    label = label_info[images_info.index(img_name)]

                    with torch.no_grad():
                        cnn_input, patch, patch_graph = NonOverlappingCropPatches(im, args=args, transforms=True)
                        cnn_input = cnn_input.unsqueeze(0)
                        patch = patch.unsqueeze(0)
                        patch_graph = patch_graph.unsqueeze(0)
                        
                        # [batch_size, image_nodes, patch_nodes, n_features]
                        patch = torch.as_tensor(patch.cuda(), dtype=torch.float32)
                        # [image_nodes, batch_size, patch_nodes, n_features]
                        patch = patch.permute([1, 0, 2, 3, 4])
                        patch = patch.reshape([-1, args.image_n_nodes, args.patch_n_nodes, args.n_features])
                        
                        # [batch_size, image_nodes, patch_nodes, patch_nodes]
                        patch_graph = torch.as_tensor(patch_graph.cuda(), dtype=torch.float32)
                        # [image_nodes, batch_size, patch_nodes, patch_nodes]
                        patch_graph = patch_graph.permute([1, 0, 2, 3, 4])
                        patch_graph = patch_graph.reshape([-1, args.image_n_nodes, args.patch_n_nodes, args.patch_n_nodes])
                        
                        # [batch_size, num_patch, 3, patch_size, patch_size]
                        cnn_input = torch.as_tensor(cnn_input.cuda(), dtype=torch.float32)
                        # [num_patch, batch_size, 3, patch_size, patch_size]
                        cnn_input = cnn_input.permute([1, 0, 2, 3, 4])
                        num_patch = np.shape(cnn_input)[0]
                        num_batch = np.shape(cnn_input)[1]
                        cnn_input = cnn_input.reshape([-1, 3, args.patch_size, args.patch_size])

                        pre = model(patch, patch_graph, cnn_input)[0]
                        pre = torch.reshape(pre, [-1]).cpu().detach().numpy()

                        pred_scores += np.mean(pre.reshape([num_patch, num_batch]), axis=0).tolist()
                        gt_scores.append(label)

            pred_scores = np.array(pred_scores)
            gt_scores = np.array(gt_scores)
            srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            plcc, _ = stats.pearsonr(pred_scores, gt_scores)
            srcc_all.append(srcc)
            plcc_all.append(plcc)

            print(str(args.model_file) + ': CSIQ Database: ', str(individual_type), ' - SRCC', srcc, ' - PLCC', plcc)

        srcc_average = np.mean(srcc_all)
        plcc_average = np.mean(plcc_all)
        print(str(individual_type), ': mean srcc: ', srcc_average, ' mean PLCC: ', plcc_average)
        end = time.time()
        print('Evaluated time is ', end - start)


if __name__ == "__main__":
    parser = ArgumentParser(description='Non-local Modeling for Image Quality Assessment') 

    parser.add_argument("--model_file", default=None, type=str, help="model file (default: models/CNNIQA-LIVE)")

    # Training parameters
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate (default: 0.0005)')
    parser.add_argument('--lrratio', default=5, type=float, help='Learning rate Ratio for GNN model (default: 10)')
    parser.add_argument('--lr_decay_ratio', default=0.50, type=float, 
                        help='learning rate multiply lr_decay_ratio (default: 0.90)')
    parser.add_argument('--lr_decay_epoch', default=5, type=int, 
                        help='Learning rate decay after lr_decay_epoch (default: 10)')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers to load training data (default: 2 X num_cores = 8)')

    # Choose Database
    parser.add_argument('--num_dis_type', default=6, type=int, help='Number of distortion types (default: TID2013: 24)')
    parser.add_argument('--database_path', default='/media/shuyuej/Projects/Dataset/', type=str, 
                        help='Database Path (default: ./dataset/)')
    parser.add_argument('--save_model', default=True, type=bool, help='Flag whether to save model (default: True)')
    parser.add_argument('--save_model_path', default='./save_model/', type=str, 
                        help='Choose to save the model or not (default: ./save_model/)')

    # SLIC superpixel Hyper-parameters
    parser.add_argument('--region_size', default=8, type=int, help='[Superpixel] region size (default: 70)')
    parser.add_argument('--ruler', default=10.0, type=float, help='[Superpixel] Ruler for slic algorithm (default: 20.0)')
    parser.add_argument('--iterate', default=10, type=int, help='[Superpixel] iterative time for slic algorithm (default: 10)')

    # GNN Hyper-parameters
    parser.add_argument('--n_features', default=3, type=int, help='Number of features per node (default: 3, w.r.t. RGB Image)')
    parser.add_argument('--hidden', default=32, type=int, help='Number of hidden units (default: 32)')
    parser.add_argument('--nb_heads', default=4, type=int, help='Number of head attentions (default: 8)')
    parser.add_argument('--threshold', default=0.70, type=float, help='Threshold for aggregating features (default: 0.70)')
    parser.add_argument('--dropout', default=0.35, type=float, help='Dropout rate (1 - keep probability) (default: 0.6)')
    parser.add_argument('--patch_n_nodes', default=60, type=int, help='Number of nodes for a image-level graph (default: 100)')
    parser.add_argument('--image_n_nodes', default=100, type=int, help='Number of nodes for a patch-level superpixels (default: 150)')

    # CNN Hyperparameters
    parser.add_argument('--patch_size', default=112, type=int, help='Image Patch Size for CNN model (default: 224)')
    parser.add_argument('--n_patches_train', default=1, type=int, help='Number of patches for CNN model (default: 1)')

    # Others
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay (default: 5 X 10^-4)')

    args = parser.parse_args()
    run(args)

