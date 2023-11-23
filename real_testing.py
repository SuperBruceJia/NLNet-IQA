#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys
import time
import warnings
from argparse import ArgumentParser

import PIL
import torch
import numpy as np

from lib.utils import *
from benchmark.database import database, distortion_type
from model.solver import GNNSolver
from model.network import Network
from lib.image_process import NonOverlappingCropPatches


def run(args):
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    print('*' * 100)
    print(' '.join(sys.argv), '\n')
    for k, v in args.__dict__.items():
        print(k, ':', v)
    print('*' * 100, '\n')

    # Load model
    model = Network(args=args).cuda()
    # Please load the model via the next line if you wanna conduct inference on CPU 
    # model.load_state_dict(torch.load(args.model_file, map_location=torch.device('cpu')))
    model.load_state_dict(torch.load(args.model_file))
    model.train(False)

    # Start time
    start = time.time()
    
    # Load Image
    im = Image.open(args.im_path)

    # Get superpixels and graph
    # Note: You can choose not to resize it!!
    height = 384
    width = 512
    img = im.resize((width, height), PIL.Image.LANCZOS)

    # Show image
    plt.figure("image")
    plt.imshow(img)
    plt.savefig('./cr7-resized.jpg')

    pred_scores = []
    gt_scores = []

    with torch.no_grad():
        # torch.Size([12, 3, 112, 112]) torch.Size([12, 100, 60, 3]) torch.Size([12, 100, 60, 60])
        cnn_input, patch, patch_graph = NonOverlappingCropPatches(img, args=args, transforms=True)
        print(cnn_input.shape, patch.shape, patch_graph.shape)

        patch = torch.as_tensor(patch.cuda(), dtype=torch.float32)  # [batch_size, image_nodes, patch_nodes, n_features]
        patch_graph = torch.as_tensor(patch_graph.cuda(), dtype=torch.float32)  # [batch_size, image_nodes, patch_nodes, patch_nodes]
        cnn_input = torch.as_tensor(cnn_input.cuda(), dtype=torch.float32)  # [batch_size, num_patch, 3, patch_size, patch_size]

        pre = model(patch, patch_graph, cnn_input)[0]
        final_pre = torch.reshape(pre, [-1]).mean().cpu().detach().numpy()

    end = time.time()
    print('The quality of this image is ', final_pre)
    print('Evaluated time is ', end - start)


if __name__ == "__main__":
    parser = ArgumentParser(description='Non-local Modeling for Image Quality Assessment')

    parser.add_argument("--im_path", default='cr7.jpg', type=str, help="image path")
    parser.add_argument("--model_file", default='save_model/CSIQ-32-4-1.pth', type=str, help="model file (default: models/CNNIQA-LIVE)")

    # Training parameters
    parser.add_argument('--exp_id', default=1, type=int, help='The k-th fold used for test (default: 1)')
    parser.add_argument('--gpu', default=0, type=int, help='Use which GPU (default: 0)')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers to load training data (default: 2 X num_cores = 8)')

    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs for training (default: 10000)')

    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate (default: 0.0005)')
    parser.add_argument('--lr_decay_ratio', default=0.50, type=float, help='learning rate multiply lr_decay_ratio (default: 0.90)')
    parser.add_argument('--lr_decay_epoch', default=20, type=int, help='Learning rate decay after lr_decay_epoch (default: 10)')

    # Choose Database
    parser.add_argument('--database', default='CSIQ', type=str, help="Choose one of the Databases")
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
    
    # CNN Hyperparameters
    parser.add_argument('--patch_size', default=112, type=int, help='Image Patch Size for CNN model (default: 224)')
    parser.add_argument('--n_patches_train', default=1, type=int, help='Number of patches for CNN model (default: 1)')

    # Others
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay (default: 5 X 10^-4)')

    args = parser.parse_args()

    # Get database info
    args.data_info, args.im_dir = database(benchmark=args.database, path=args.database_path)
    args.num_dis_type = distortion_type(database=args.database)

    run(args)
