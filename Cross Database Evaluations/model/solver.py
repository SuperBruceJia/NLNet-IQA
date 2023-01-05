# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from model.network import Network
from data_process.load_data import data_loader
from lib.utils import evaluation_criteria


class GNNSolver(object):
    """Solver for training and testing hyperIQA"""

    def __init__(self, args):
        self.args = args

        self.epochs = args.epochs
        self.batch_size = args.batch_size

        self.lr = args.lr
        self.lr_decay_ratio = args.lr_decay_ratio
        self.lr_decay_epoch = args.lr_decay_epoch
        self.weight_decay = args.weight_decay

        self.save_model_path = args.save_model_path
        self.train_database = args.train_database
        self.test_database = args.test_database

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_n_nodes = args.image_n_nodes
        self.patch_n_nodes = args.patch_n_nodes
        self.patch_size = args.patch_size

        self.n_features = args.n_features
        self.hidden = args.hidden
        self.nb_heads = args.nb_heads
        self.dropout = args.dropout

        self.n_patches_train = args.n_patches_train

        # Choose which model to train
        self.model = Network(args=args).cuda()

        print('*' * 100)
        print(self.model)
        print('*' * 100, '\n')

        paras = self.model.parameters()

        print('*' * 100)
        print('The trained parameters are as follows: \n')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, ', with shape: ', np.shape(param))
        print('*' * 100, '\n')

        self.model.train(True)

        # Loss Functions
        self.quality_loss = nn.SmoothL1Loss().cuda()
        self.quality_rank_loss = nn.SmoothL1Loss().cuda()
        self.dis_type_loss = nn.CrossEntropyLoss().cuda()

        # Optimizer
        self.solver = optim.Adam(params=filter(lambda p: p.requires_grad, paras), lr=self.lr, weight_decay=self.weight_decay)

        # Learning rate decay scheduler - Every lr_decay_epoch epochs, LR * lr_decay_ratio
        self.scheduler = lr_scheduler.StepLR(optimizer=self.solver, step_size=self.lr_decay_epoch, gamma=self.lr_decay_ratio)

        # Get training, and testing data
        self.train_loader, self.test_loader = data_loader(args)

    def train(self):
        """Training"""
        test_srcc = 0.0
        test_plcc = 0.0
        test_krcc = 0.0
        test_rmse = 0.0
        test_mae = 0.0

        print('Epoch TRAINING Loss\t '
              'TRAINING SRCC PLCC KRCC MSE MAE OR\t '
              'TESTING SRCC PLCC KRCC MSE MAE OR\t ')

        for t in range(1, self.epochs + 1):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for i, (patch, patch_graph, cnn_input, label, dis_type) in enumerate(self.train_loader):
                patch = torch.as_tensor(patch.cuda(), dtype=torch.float32)  # [batch_size, num_patch, image_nodes, patch_nodes, n_features]
                patch = patch.permute([1, 0, 2, 3, 4])  # [num_patch, batch_size, image_nodes, patch_nodes, n_features]
                patch = patch.reshape([-1, self.image_n_nodes, self.patch_n_nodes, self.n_features])

                patch_graph = torch.as_tensor(patch_graph.cuda(), dtype=torch.float32)  # [batch_size, num_patch, image_nodes, patch_nodes, patch_nodes]
                patch_graph = patch_graph.permute([1, 0, 2, 3, 4])  # [num_patch, batch_size, image_nodes, patch_nodes, patch_nodes]
                patch_graph = patch_graph.reshape([-1, self.image_n_nodes, self.patch_n_nodes, self.patch_n_nodes])

                cnn_input = torch.as_tensor(cnn_input.cuda(), dtype=torch.float32)  # [batch_size, num_patch, 3, patch_size, patch_size]
                cnn_input = cnn_input.permute([1, 0, 2, 3, 4])  # [num_patch, batch_size, 3, patch_size, patch_size]
                num_patch = np.shape(cnn_input)[0]
                num_batch = np.shape(cnn_input)[1]
                cnn_input = cnn_input.reshape([-1, 3, self.patch_size, self.patch_size])

                label = torch.as_tensor(label.cuda(), dtype=torch.float32)  # [batch_size, 1]
                label = torch.reshape(label, [-1])  # [batch_size]

                dis_type = torch.as_tensor(dis_type.cuda(), dtype=torch.long)  # [batch_size, 1]
                dis_type = torch.reshape(dis_type, [-1])  # [batch_size]
                dis_type = dis_type.repeat(num_patch)

                self.solver.zero_grad()

                pre, type_pre = self.model(patch, patch_graph, cnn_input)
                pre = torch.reshape(pre, [-1])

                temp_scores = pre.cpu().detach().numpy().copy()
                pred_scores += np.mean(temp_scores.reshape([num_patch, num_batch]), axis=0).tolist()
                gt_scores += label.cpu().tolist()

                # Overall Loss
                rank_loss = 0.0
                rank_id = [(i, j) for i in range(len(pre)) for j in range(len(pre)) if i != j and i <= j]
                for i in range(len(rank_id)):
                    pre_1 = pre[rank_id[i][0]]
                    pre_2 = pre[rank_id[i][1]]
                    label_1 = label[rank_id[i][0]]
                    label_2 = label[rank_id[i][1]]
                    rank_loss += self.quality_rank_loss(pre_1 - pre_2, label_1 - label_2)

                if len(pre) != 1:
                    rank_loss /= (len(pre) * (len(pre) - 1) / 2)

                # Quality Regression Loss
                quality_loss = self.quality_loss(pre, label)
                dis_type_loss = self.dis_type_loss(type_pre, dis_type)
                loss = 100 * quality_loss + rank_loss + dis_type_loss

                epoch_loss.append(loss.detach())
                loss.backward()
                self.solver.step()

            train_srcc, train_plcc, train_krcc, train_rmse, train_mae \
                = evaluation_criteria(pre=pred_scores, label=gt_scores)

            # Get the performance on the testing set w.r.t. the best validation performance
            test_srcc, test_plcc, test_krcc, test_rmse, test_mae = self.validate_test(self.test_loader)

            torch.save(self.model.state_dict(),
                       self.save_model_path
                       + 'train-' + self.train_database
                       + '-test-' + self.test_database
                       + '.pth')

            print("%d, %4.4f, || "
                  "%4.4f, %4.4f, %4.4f, %4.4f, %4.4f, || "
                  "%4.4f, %4.4f, %4.4f, %4.4f,  %4.4f"
                  % (t, sum(epoch_loss) / len(epoch_loss),
                     train_srcc, train_plcc, train_krcc, train_rmse, train_mae,
                     test_srcc, test_plcc, test_krcc, test_rmse, test_mae))

            # Learning rate decay
            self.scheduler.step()

        return test_srcc, test_plcc, test_krcc, test_rmse, test_mae

    def validate_test(self, data):
        """Validation and Testing"""
        self.model.train(False)

        pred_scores = []
        gt_scores = []

        with torch.no_grad():
            for patch, patch_graph, cnn_input, label, _ in data:
                patch = torch.as_tensor(patch.cuda(), dtype=torch.float32)  # [batch_size, image_nodes, patch_nodes, n_features]
                patch = patch.permute([1, 0, 2, 3, 4])  # [image_nodes, batch_size, patch_nodes, n_features]
                patch = patch.reshape([-1, self.image_n_nodes, self.patch_n_nodes, self.n_features])

                patch_graph = torch.as_tensor(patch_graph.cuda(), dtype=torch.float32)  # [batch_size, image_nodes, patch_nodes, patch_nodes]
                patch_graph = patch_graph.permute([1, 0, 2, 3, 4])  # [image_nodes, batch_size, patch_nodes, patch_nodes]
                patch_graph = patch_graph.reshape([-1, self.image_n_nodes, self.patch_n_nodes, self.patch_n_nodes])

                cnn_input = torch.as_tensor(cnn_input.cuda(), dtype=torch.float32)  # [batch_size, num_patch, 3, patch_size, patch_size]
                cnn_input = cnn_input.permute([1, 0, 2, 3, 4])  # [num_patch, batch_size, 3, patch_size, patch_size]
                num_patch = np.shape(cnn_input)[0]
                num_batch = np.shape(cnn_input)[1]
                cnn_input = cnn_input.reshape([-1, 3, self.patch_size, self.patch_size])

                label = torch.as_tensor(label.cuda(), dtype=torch.float32)  # [batch_size, 1]
                label = torch.reshape(label, [-1])  # [batch_size]

                pre = self.model(patch, patch_graph, cnn_input)[0]
                pre = torch.reshape(pre, [-1]).cpu().detach().numpy()

                pred_scores += np.mean(pre.reshape([num_patch, num_batch]), axis=0).tolist()
                gt_scores += label.cpu().tolist()

        srcc, plcc, krcc, rmse, mae \
            = evaluation_criteria(pre=pred_scores, label=gt_scores)

        self.model.train(True)

        return srcc, plcc, krcc, rmse, mae
