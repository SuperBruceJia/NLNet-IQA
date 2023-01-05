# -*- coding: utf-8 -*-

import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from model.layers import Attention
from lib.utils import vec_l2_norm, bilinear_pool, gaussian_prior, L2pooling, GDN, node_norm, group_norm


class Network(nn.Module):
    """NLNet Network"""
    def __init__(self, args):
        super(Network, self).__init__()

        self.nfeat = args.n_features
        self.nhid = args.hidden
        self.nheads = args.nb_heads

        self.threshold = args.threshold
        self.dropout = args.dropout

        self.image_n_nodes = args.image_n_nodes
        self.patch_n_nodes = args.patch_n_nodes

        self.num_dis_type = args.num_dis_type

        ############################################################################################################################################
        # Pre-trained VGGNet 16
        self.cnn_backbone = models.vgg16(pretrained=True).features

        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        for x in range(0, 5):
            self.stage1.add_module(str(x), self.cnn_backbone._modules[str(x)])

        for x in range(5, 10):
            self.stage2.add_module(str(x), self.cnn_backbone._modules[str(x)])

        for x in range(10, 17):
            self.stage3.add_module(str(x), self.cnn_backbone._modules[str(x)])

        for x in range(17, 24):
            self.stage4.add_module(str(x), self.cnn_backbone._modules[str(x)])

        for x in range(24, 31):
            self.stage5.add_module(str(x), self.cnn_backbone._modules[str(x)])

        ############################################################################################################################################
        self.patch_attentions1 = Attention(number_nodes=self.patch_n_nodes,
                                           in_features=self.nfeat,
                                           out_features=self.nhid,
                                           nheads=self.nheads,
                                           threshold=self.threshold,
                                           dropout=self.dropout)
        self.image_attentions1 = Attention(number_nodes=self.image_n_nodes,
                                           in_features=self.nheads * self.nhid * 2,
                                           out_features=self.nhid * 2,
                                           nheads=self.nheads,
                                           threshold=self.threshold,
                                           dropout=self.dropout)
        self.image_attentions2 = Attention(number_nodes=self.image_n_nodes,
                                           in_features=self.nhid * self.nheads * 2,
                                           out_features=self.nhid * 2,
                                           nheads=self.nheads,
                                           threshold=self.threshold,
                                           dropout=self.dropout)
        self.image_attentions3 = Attention(number_nodes=self.image_n_nodes,
                                           in_features=self.nhid * self.nheads * 2,
                                           out_features=self.nhid * 2,
                                           nheads=self.nheads,
                                           threshold=self.threshold,
                                           dropout=self.dropout)

        ############################################################################################################################################
        # FC For Distortion Type Classification
        self.dis_type_linear1 = nn.Linear((64 + 128 + 256 + 512 + 512) * 2 + self.nheads * self.nhid * 16, 512)
        self.dis_type_linear2 = nn.Linear(512, self.num_dis_type)

        # FC For Quality Score Regression
        self.quality_linear1 = nn.Linear((64 + 128 + 256 + 512 + 512) * 2 + self.nheads * self.nhid * 16, 512)
        self.quality_linear2 = nn.Linear(512, 1)

    def patch_extract_features(self, x, adj):
        """
        :param x: [batch_size, patch_nodes, n_features]
        :param adj: [batch_size, patch_nodes, patch_nodes]
        """
        x1 = self.patch_attentions1(x, adj)  # [self.batch, self.patch_n_nodes, self.nheads * self.nhid]
        x1 = node_norm(x=x1, p=2)

        x1_mean = torch.mean(x1, dim=1, keepdim=True)  # [self.batch, 1, self.nheads * self.nhid]
        x1_std = torch.std(x1, dim=1, keepdim=True)  # [self.batch, 1, self.nheads * self.nhid]

        out = torch.cat([x1_mean, x1_std], dim=1)  # [self.batch, 2, self.nheads * self.nhid]
        out = out.view(-1, 1, self.nheads * self.nhid * 2)  # [self.batch, self.nheads * self.nhid * 2]

        return out

    def gnn_extract_features(self, x, adj):
        """
        :param x: [batch_size, image_n_nodes, self.nheads * self.nhid * 2]
        :param adj: [batch_size, image_n_nodes, image_n_nodes]
        """
        alpha = 0.2
        x_mean = torch.mean(x, dim=1, keepdim=True)  # [self.batch, 1, self.nheads * self.nhid * 2]
        x_std = torch.std(x, dim=1, keepdim=True)  # [self.batch, 1, self.nheads * self.nhid * 2]

        x1 = self.image_attentions1(x, adj)  # [self.batch, self.image_n_nodes, self.nhid * self.nheads * 2]
        x1 = node_norm(x=x1, p=2)
        x1 = F.elu((1 - alpha) * x1 + alpha * x)
        x1_mean = torch.mean(x1, dim=1, keepdim=True)  # [self.batch, 1, self.nhid * self.nheads * 2]
        x1_std = torch.std(x1, dim=1, keepdim=True)  # [self.batch, 1, self.nhid * self.nheads * 2]

        x2 = self.image_attentions2(x1, adj)  # [self.batch, self.image_n_nodes, self.nhid * self.nheads * 2]
        x2 = node_norm(x=x2, p=2)
        x2 = F.elu((1 - alpha) * x2 + alpha * x)
        x2_mean = torch.mean(x2, dim=1, keepdim=True)  # [self.batch, 1, self.nhid * self.nheads * 2]
        x2_std = torch.std(x2, dim=1, keepdim=True)  # [self.batch, 1, self.nhid * self.nheads * 2]

        x3 = self.image_attentions3(x2, adj)  # [self.batch, self.image_n_nodes, self.nhid * self.nheads * 2]
        x3 = node_norm(x=x3, p=2)
        x3 = F.elu((1 - alpha) * x3 + alpha * x)
        x3_mean = torch.mean(x3, dim=1, keepdim=True)  # [self.batch, 1, self.nhid * self.nheads * 2]
        x3_std = torch.std(x3, dim=1, keepdim=True)  # [self.batch, 1, self.nhid * self.nheads * 2]

        # [self.batch, self.nheads * self.nhid * 16]
        out = torch.cat([x_mean.squeeze(dim=1), x_std.squeeze(dim=1),
                         x1_mean.squeeze(dim=1), x1_std.squeeze(dim=1),
                         x2_mean.squeeze(dim=1), x2_std.squeeze(dim=1),
                         x3_mean.squeeze(dim=1), x3_std.squeeze(dim=1)], dim=1)

        return out

    def build_image_graph(self, feature):
        """
        :param feature: [batch_size, image_n_nodes, self.nhid * self.nheads * 2]
        :return image_graph: [batch_size, image_n_nodes, image_n_nodes]
        """
        image_graph = torch.cat([torch.cosine_similarity(x1=feature[:, i, :].unsqueeze(dim=1),
                                                         x2=feature, dim=2).unsqueeze(dim=2)
                                 for i in range(self.image_n_nodes)], dim=2)

        return image_graph

    def cnn_extract_features(self, cnn_input):
        cnn_feature = self.stage1(cnn_input)
        stage1_feat = cnn_feature.view(cnn_feature.size(0), 64, 56 * 56)  # [self.batch, 64, 224, 224]
        stage1_feat_mean = torch.mean(stage1_feat, dim=2, keepdim=True)
        stage1_feat_std = torch.std(stage1_feat, dim=2, keepdim=True)

        cnn_feature = self.stage2(cnn_feature)
        stage2_feat = cnn_feature.view(cnn_feature.size(0), 128, 28 * 28)  # [self.batch, 128, 112, 112]
        stage2_feat_mean = torch.mean(stage2_feat, dim=2, keepdim=True)
        stage2_feat_std = torch.std(stage2_feat, dim=2, keepdim=True)

        cnn_feature = self.stage3(cnn_feature)
        stage3_feat = cnn_feature.view(cnn_feature.size(0), 256, 14 * 14)  # [self.batch, 256, 56, 56]
        stage3_feat_mean = torch.mean(stage3_feat, dim=2, keepdim=True)
        stage3_feat_std = torch.std(stage3_feat, dim=2, keepdim=True)

        cnn_feature = self.stage4(cnn_feature)
        stage4_feat = cnn_feature.view(cnn_feature.size(0), 512, 7 * 7)  # [self.batch, 512, 28, 28]
        stage4_feat_mean = torch.mean(stage4_feat, dim=2, keepdim=True)
        stage4_feat_std = torch.std(stage4_feat, dim=2, keepdim=True)

        cnn_feature = self.stage5(cnn_feature)
        stage5_feat = cnn_feature.view(cnn_feature.size(0), 512, 3 * 3)  # [self.batch, 512, 14, 14]
        stage5_feat_mean = torch.mean(stage5_feat, dim=2, keepdim=True)
        stage5_feat_std = torch.std(stage5_feat, dim=2, keepdim=True)

        # Output dim: [self.batch, 64 + 128 + 256 + 512 + 512, 1]
        cnn_cat_mean = torch.cat([stage1_feat_mean, stage2_feat_mean, stage3_feat_mean,
                                  stage4_feat_mean, stage5_feat_mean], dim=1)
        # Output dim: [self.batch, 64 + 128 + 256 + 512 + 512, 1]
        cnn_cat_std = torch.cat([stage1_feat_std, stage2_feat_std, stage3_feat_std,
                                 stage4_feat_std, stage5_feat_std], dim=1)

        # Output dim: [self.batch, (64 + 128 + 256 + 512 + 512) * 2]
        out = torch.cat([cnn_cat_mean, cnn_cat_std], dim=2)
        out = out.view(-1, (64 + 128 + 256 + 512 + 512) * 2)

        return out

    def forward(self, patch_input, patch_graph, cnn_input):
        # patch_input: [self.batch_size, self.image_nodes, self.patch_nodes, self.n_features]
        # patch_graph: [self.batch_size, self.image_nodes, self.patch_nodes, self.patch_nodes]
        patch_input = patch_input.permute([1, 0, 2, 3])  # [image_nodes, batch_size, patch_nodes, n_features]
        patch_graph = patch_graph.permute([1, 0, 2, 3])  # [image_nodes, batch_size, patch_nodes, patch_nodes]

        # Patch-level aggregate features [self.batch_size, self.image_n_nodes, self.nheads * self.nhid * 2]
        image_feature = torch.cat([self.patch_extract_features(patch_input[i], patch_graph[i])
                                   for i in range(patch_input.size(0))], dim=1)

        # Make image graph
        image_graph = self.build_image_graph(image_feature)

        # Image-level GNN
        gnn_feature = self.gnn_extract_features(image_feature, image_graph)

        # CNN Feature
        cnn_feature = self.cnn_extract_features(cnn_input)

        # Future Fusion
        fused_feature = torch.cat([gnn_feature, cnn_feature], dim=1)

        # For Distortion Type Classification
        dis_type_out = F.elu(self.dis_type_linear1(fused_feature))
        dis_type_out = F.softmax(self.dis_type_linear2(dis_type_out), dim=1)

        # For Quality Prediction
        quality_out = F.elu(self.quality_linear1(fused_feature))
        quality_out = self.quality_linear2(quality_out)

        return quality_out, dis_type_out
