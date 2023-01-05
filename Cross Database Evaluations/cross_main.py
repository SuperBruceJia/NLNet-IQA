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
from benchmark.database import cross_database, distortion_type
from model.solver import GNNSolver


def run(args):
    """
    Run the program
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cudnn.benchmark = True
    print('Current training database is ', args.train_database, '\n')
    print('Current testing database is ', args.test_database, '\n')

    print('*' * 100)
    print(' '.join(sys.argv), '\n')
    for k, v in args.__dict__.items():
        print(k, ':', v)
    print('*' * 100, '\n')

    solver = GNNSolver(args=args)
    srcc, plcc, krcc, rmse, mae = solver.train()
    print('The best testing SRCC: %4.4f, PLCC: %4.4f, KRCC: %4.4f, MSE: %4.4f, MAE: %4.4f \n\n\n\n'
          % (srcc, plcc, krcc, rmse, mae))


if __name__ == "__main__":
    parser = ArgumentParser(description='Non-local Modeling for Image Quality Assessment')

    # Training parameters
    parser.add_argument('--gpu', default=0, type=int, help='Use which GPU (default: 0)')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers to load training data (default: 2 X num_cores = 8)')

    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs for training (default: 10000)')

    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate (default: 0.0005)')
    parser.add_argument('--lr_decay_ratio', default=0.50, type=float, help='learning rate multiply lr_decay_ratio (default: 0.90)')
    parser.add_argument('--lr_decay_epoch', default=20, type=int, help='Learning rate decay after lr_decay_epoch (default: 10)')

    # Choose Database
    parser.add_argument('--train_database', default='CSIQ', type=str)
    parser.add_argument('--test_database', default='LIVE', type=str)
    parser.add_argument('--database_path', default='/media/shuyuej/Projects/Dataset/', type=str, help='Database Path (default: ./dataset/)')
    parser.add_argument('--save_model_path', default='./save_model/', type=str, help='Choose to save the model or not (default: ./save_model/)')

    # SLIC superpixel Hyper-parameters
    parser.add_argument('--region_size', default=8, type=int, help='[Superpixel] region size (default: 70)')
    parser.add_argument('--ruler', default=10.0, type=float, help='[Superpixel] Ruler for slic algorithm (default: 20.0)')
    parser.add_argument('--iterate', default=10, type=int, help='[Superpixel] iterative time for slic algorithm (default: 10)')

    # GNN Hyper-parameters
    parser.add_argument('--hidden', default=32, type=int, help='Number of hidden units (default: 32)')
    parser.add_argument('--nb_heads', default=4, type=int, help='Number of head attentions (default: 8)')
    parser.add_argument('--threshold', default=0.70, type=float, help='Threshold for aggregating features (default: 0.70)')
    parser.add_argument('--dropout', default=0.35, type=float, help='Dropout rate (1 - keep probability) (default: 0.6)')
    parser.add_argument('--n_features', default=3, type=int, help='Number of features per node (default: 3, w.r.t. RGB Image)')
    parser.add_argument('--patch_n_nodes', default=60, type=int, help='Number of nodes for a image-level graph (default: 100)')
    parser.add_argument('--image_n_nodes', default=100, type=int, help='Number of nodes for a patch-level superpixels (default: 150)')
    
    # CNN Hyper-parameters
    parser.add_argument('--patch_size', default=112, type=int, help='Image Patch Size for CNN model (default: 224)')
    parser.add_argument('--n_patches_train', default=1, type=int, help='Number of patches for CNN model (default: 1)')

    # Others
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay (default: 5 X 10^-4)')

    args = parser.parse_args()

    # Get database
    args.train_info, args.train_im_dir, args.train_ref_dir, args.test_info, args.test_im_dir, args.test_ref_dir \
        = cross_database(train=args.train_database, test=args.test_database, path=args.database_path)
    args.num_dis_type = distortion_type(database=args.train_database)

    run(args)

