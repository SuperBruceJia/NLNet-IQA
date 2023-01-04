#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys
import warnings
import numpy as np
from argparse import ArgumentParser

from lib.utils import *
from benchmark.database import database, distortion_type
from model.solver import GNNSolver


def run(args):
    """
    Run the program
    """
    # CUDA Configurations
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cudnn.benchmark = True
    print('Current Train/Validation/Testing database is', args.database, '\n')

    # Print Hyper-parameter Configurations
    print('*' * 100)
    print(' '.join(sys.argv), '\n')
    for k, v in args.__dict__.items():
        print(k, ':', v)
    print('*' * 100, '\n')

    # 10 random splits of the reference indices by setting random seed from 1 to 10
    k_fold_cv = 10
    srcc_all = np.zeros(k_fold_cv, dtype=float)
    plcc_all = np.zeros(k_fold_cv, dtype=float)
    krcc_all = np.zeros(k_fold_cv, dtype=float)
    rmse_all = np.zeros(k_fold_cv, dtype=float)
    mae_all = np.zeros(k_fold_cv, dtype=float)
    or_all = np.zeros(k_fold_cv, dtype=float)

    index = 0
    for id in range(1, k_fold_cv + 1):
        args.exp_id = id

        print('This is the %d validate and test regarding 10-fold cross-validation\n' % args.exp_id)
        # Initialize the model
        solver = GNNSolver(args=args)
        # Train the model
        srcc_all[index], plcc_all[index], krcc_all[index], \
            rmse_all[index], mae_all[index], or_all[index] = solver.train()

        print('The best testing SRCC: %4.4f, PLCC: %4.4f, KRCC: %4.4f, MSE: %4.4f, MAE: %4.4f, OR: %4.4f \n\n\n\n'
              % (srcc_all[index], plcc_all[index], krcc_all[index], rmse_all[index], mae_all[index], or_all[index]))

        index += 1

    # The median SRCC/PLCC performances on the testing set are reported
    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)
    krcc_med = np.median(krcc_all)
    rmse_med = np.median(rmse_all)
    mae_med = np.median(mae_all)
    or_med = np.median(or_all)
    print(str(k_fold_cv), 'random-split running result: \n')
    print('Testing median SRCC %4.4f, median PLCC %4.4f, median KRCC %4.4f, median MSE %4.4f,'
          'median MAE %4.4f, median OR %4.4f \n\n\n\n'
          % (srcc_med, plcc_med, krcc_med, rmse_med, mae_med, or_med))


if __name__ == "__main__":
    parser = ArgumentParser(description='Non-local Modeling for Image Quality Assessment')

    # Training parameters
    parser.add_argument('--exp_id', default=1, type=int, help='The k-th random split running.')
    parser.add_argument('--gpu', default=0, type=int, help='Use which GPU (default: 0)')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of workers to load training data (default: 2 X num_cores = 8)')

    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs for training (default: 100)')

    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate (default: 1e^-4)')
    parser.add_argument('--lr_decay_ratio', default=0.50, type=float,
                        help='learning rate multiply lr_decay_ratio (default: 0.50)')
    parser.add_argument('--lr_decay_epoch', default=20, type=int,
                        help='Learning rate decay after lr_decay_epoch (default: 20)')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay (default: 5 X 10^-4)')

    # Choose Database
    parser.add_argument('--database', default='TID2013', type=str,
                        help="Choose one of the Databases: LIVE, CSIQ, TID2013, and KADID")
    parser.add_argument('--database_path', default='Dataset/', type=str,
                        help='Database Path (default: Dataset/)')
    parser.add_argument('--save_model_path', default='./save_model/', type=str,
                        help='Choose to save the model or not (default: ./save_model/)')

    # SLIC superpixel Hyper-parameters
    parser.add_argument('--region_size', default=8, type=int, help='[Superpixel] region size (default: 70)')
    parser.add_argument('--ruler', default=10.0, type=float,
                        help='[Superpixel] Ruler for slic algorithm (default: 10.0)')
    parser.add_argument('--iterate', default=10, type=int,
                        help='[Superpixel] iterative time for slic algorithm (default: 10)')

    # GNN Hyper-parameters
    parser.add_argument('--hidden', default=32, type=int, help='Number of hidden units (default: 32)')
    parser.add_argument('--nb_heads', default=4, type=int, help='Number of head attentions (default: 8)')
    parser.add_argument('--threshold', default=0.70, type=float,
                        help='Threshold for aggregating features (default: 0.70)')
    parser.add_argument('--dropout', default=0.35, type=float,
                        help='Dropout rate (1 - keep probability) (default: 0.35)')
    parser.add_argument('--n_features', default=3, type=int,
                        help='Number of features per node (default: 3, w.r.t. RGB Image)')
    parser.add_argument('--patch_n_nodes', default=60, type=int,
                        help='Number of nodes for a image-level graph (default: 60)')
    parser.add_argument('--image_n_nodes', default=100, type=int,
                        help='Number of nodes for a patch-level superpixels (default: 100)')

    # CNN Hyperparameters
    parser.add_argument('--patch_size', default=112, type=int, help='Image Patch Size for CNN model (default: 224)')
    parser.add_argument('--n_patches_train', default=1, type=int, help='Number of patches for CNN model (default: 1)')
    
    args = parser.parse_args()

    # Get database info
    args.data_info, args.im_dir = database(benchmark=args.database, path=args.database_path)
    args.num_dis_type = distortion_type(database=args.database)

    run(args)
