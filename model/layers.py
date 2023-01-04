# -*- coding: utf-8 -*-

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, number_nodes, in_features, out_features, nheads, threshold, dropout):
        super(Attention, self).__init__()

        self.number_nodes = number_nodes
        self.in_features = in_features
        self.out_features = out_features
        self.nheads = nheads

        self.dropout = dropout
        self.threshold = threshold

        self.Wh1_bias = nn.Parameter(torch.empty(size=(nheads, number_nodes, 1)))
        nn.init.xavier_uniform_(self.Wh1_bias.data, gain=1.414)

        self.Wh2_bias = nn.Parameter(torch.empty(size=(nheads, number_nodes, 1)))
        nn.init.xavier_uniform_(self.Wh2_bias.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(nheads, 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.a_bias = nn.Parameter(torch.empty(size=(nheads, number_nodes, number_nodes)))
        nn.init.xavier_uniform_(self.a_bias.data, gain=1.414)

        self.attention_bias = nn.Parameter(torch.empty(size=(nheads, number_nodes, out_features)))
        nn.init.xavier_uniform_(self.attention_bias.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)

        self.conv_1d = [nn.Conv1d(self.in_features, self.out_features, kernel_size=1, stride=1, padding=0)
                        for _ in range(nheads)]
        for i, conv in enumerate(self.conv_1d):
            self.add_module('conv_{}'.format(i), conv)

    def forward(self, h, adj):
        """
        :param h: (batch_zize, number_nodes, in_features)
        :param adj: (batch_size, number_nodes, number_nodes)
        :return: (batch_zize, number_nodes, self.out_features * nheads)
        """
        # Linear transformation
        h = F.dropout(h, self.dropout, training=self.training)
        Wh = torch.cat([conv(h.permute([0, 2, 1])).unsqueeze(dim=1) for conv in self.conv_1d], dim=1)
        Wh = Wh.permute([0, 1, 3, 2])

        # Shared attention Mechanism (batch_zize, nheads, number_nodes, number_nodes)
        e = self.shared_attention(Wh)

        # Mask: (batch_zize, nheads, number_nodes, number_nodes)
        zero_vec = -10e9 * torch.ones_like(e)

        # Mask: (batch_zize, nheads, number_nodes, number_nodes), adj: (batch_size, 1, number_nodes, number_nodes)
        adj = adj.unsqueeze(dim=1)
        attention = torch.where(adj >= self.threshold, e, zero_vec)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Mask attention (batch_zize, nheads, number_nodes, number_nodes)
        attention_visualize = F.softmax(attention, dim=3)
        attention = F.dropout(attention_visualize, self.dropout, training=self.training)
        Wh = F.dropout(Wh, self.dropout, training=self.training)

        # Apply masked attention
        # (batch_zize, nheads, number_nodes, number_nodes) X (batch_zize, nheads, number_nodes, out_features)
        # -> (batch_zize, nheads, number_nodes, out_features)
        h_prime = F.elu(torch.einsum('bhnn,bhno->bhno', attention, Wh) + self.attention_bias)
        # h_prime = F.dropout(h_prime, self.dropout, training=self.training)

        # (batch_zize, nheads, number_nodes, out_features)
        # (batch_zize, number_nodes, nheads, out_features) -> (batch_zize, number_nodes, nheads * out_features)
        h_prime = h_prime.permute(0, 2, 1, 3).reshape(h_prime.shape[0],
                                                      h_prime.shape[2],
                                                      h_prime.shape[1] * h_prime.shape[3])

        return h_prime

    def shared_attention(self, Wh):
        """
        # Wh.shape (batch_zize, nheads, number_nodes, out_feature)
        # self.a.shape (nheads, 2 * out_feature, 1)
        # Wh1 and Wh2.shape (batch_zize, nheads, number_nodes, 1)
        # out.shape (batch_zize, nheads, number_nodes, number_nodes)
        """
        # (batch_zize, nheads, number_nodes, out_feature) X (nheads, out_feature, 1)
        # -> (batch_zize, nheads, number_nodes, 1)
        Wh1 = torch.einsum('bhno,hol->bhnl', Wh, self.a[:, :self.out_features, :]) + self.Wh1_bias
        Wh1 = F.dropout(Wh1, self.dropout, training=self.training)

        Wh2 = torch.einsum('bhno,hol->bhln', Wh, self.a[:, self.out_features:, :]) + self.Wh2_bias
        Wh2 = F.dropout(Wh2, self.dropout, training=self.training)

        # broadcast add (batch_zize, nheads, number_nodes, 1) + (batch_zize, nheads, 1, number_nodes)
        # -> (batch_zize, nheads, number_nodes, number_nodes)
        out = self.leakyrelu(Wh1 + Wh2) + self.a_bias
        out = F.dropout(out, self.dropout, training=self.training)

        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) \
            + ' -> ' + str(self.out_features * self.nheads) + ')'
