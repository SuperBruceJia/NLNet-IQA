#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from sys import stdout
import shutil
import random
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
from scipy.ndimage.filters import convolve
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Function


def evaluation_criteria(pre, label):
    pre = np.array(pre)
    label = np.array(label)

    srcc = stats.spearmanr(pre, label)[0]
    plcc = stats.pearsonr(pre, label)[0]
    krcc = stats.stats.kendalltau(pre, label)[0]
    rmse = np.sqrt(((pre - label) ** 2).mean())
    mae = np.abs((pre - label)).mean()

    return srcc, plcc, krcc, rmse, mae


def mos_rescale(mos, min_val, max_val, scale_min=0, scale_max=1):
    mos = scale_min + (mos - min_val) * ((scale_max - scale_min) / (max_val - min_val))

    return mos


def ranking_loss(pre, label, loss):
    rank_loss = 0.0
    rank_id = [(i, j) for i in range(len(pre)) for j in range(len(pre)) if i != j and i <= j]
    for i in range(len(rank_id)):
        pre_1 = pre[rank_id[i][0]]
        pre_2 = pre[rank_id[i][1]]
        label_1 = label[rank_id[i][0]]
        label_2 = label[rank_id[i][1]]
        rank_loss += loss(pre_1 - pre_2, label_1 - label_2)

    if len(pre) != 1:
        rank_loss /= (len(pre) * (len(pre) - 1) / 2)

    return rank_loss


def relative_ranking_loss(pre, label):
    # Relative Ranking Loss
    sort_index = [x for _, x in sorted(zip(pre, list(range(len(pre)))), reverse=True)]
    high_pre = pre[sort_index[0]]
    second_high_pre = pre[sort_index[1]]
    low_pre = pre[sort_index[-1]]
    second_low_pre = pre[sort_index[-2]]

    high_label = label[sort_index[0]]
    second_high_label = label[sort_index[1]]
    low_label = label[sort_index[-1]]
    second_low_label = label[sort_index[-2]]

    margin1 = second_high_label - low_label
    margin2 = high_label - second_low_label

    triplet_loss_1 = abs(high_pre - second_high_pre) - abs(high_pre - low_pre) + margin1
    triplet_loss_2 = abs(second_low_pre - low_pre) - abs(high_pre - low_pre) + margin2

    if triplet_loss_1 <= 0:
        triplet_loss_1 = 0

    if triplet_loss_2 <= 0:
        triplet_loss_2 = 0

    rank_loss = triplet_loss_1 + triplet_loss_2

    return rank_loss


def pseudo_huber_loss(pre, label, delta):

    # loss = (delta ** 2) * (torch.sqrt(1 + torch.square((pre - label) / (delta + 1e-8))) - 1)
    loss = (delta ** 2) * ((1 + ((pre - label) / (delta + 1e-8)) ** 2) ** (1 / 2) - 1)

    return loss


k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
kern = k / k.sum()


def local_normalize(img, num_ch=1, const=127.0):
    if num_ch == 1:
        mu = convolve(img[:, :, 0], kern, mode='nearest')
        mu_sq = mu * mu
        im_sq = img[:, :, 0] * img[:, :, 0]
        tmp = convolve(im_sq, kern, mode='nearest') - mu_sq
        sigma = np.sqrt(np.abs(tmp))
        structdis = (img[:, :, 0] - mu) / (sigma + const)

        # Rescale within 0 and 1
        # structdis = (structdis + 3) / 6
        structdis = 2. * structdis / 3.
        norm = structdis[:, :, None]
    elif num_ch > 1:
        norm = np.zeros(img.shape, dtype='float32')
        for ch in range(num_ch):
            mu = convolve(img[:, :, ch], kern, mode='nearest')
            mu_sq = mu * mu
            im_sq = img[:, :, ch] * img[:, :, ch]
            tmp = convolve(im_sq, kern, mode='nearest') - mu_sq
            sigma = np.sqrt(np.abs(tmp))
            structdis = (img[:, :, ch] - mu) / (sigma + const)

            # Rescale within 0 and 1
            # structdis = (structdis + 3) / 6
            structdis = 2. * structdis / 3.
            norm[:, :, ch] = structdis

    return norm


class LowerBound(Function):
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size()) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 device,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.FloatTensor([reparam_offset])

        self.build(ch, torch.device(device))

    def build(self, ch, device):
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** .5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta.to(device))

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma.to(device))
        self.pedestal = self.pedestal.to(device)

    def forward(self, inputs):
        device_id = inputs.device.index

        beta = self.beta.to(device_id)
        gamma = self.gamma.to(device_id)
        pedestal = self.pedestal.to(device_id)

        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound()(beta, self.beta_bound)
        beta = beta ** 2 - pedestal

        # Gamma bound and reparam
        gamma = LowerBound()(gamma, self.gamma_bound)
        gamma = gamma ** 2 - pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()

        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels

        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, input):
        input = input ** 2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])

        return (out + 1e-12).sqrt()


class group_norm(torch.nn.Module):
    def __init__(self, dim_to_norm=None, dim_hidden=16, num_nodes=None, num_groups=None, skip_weight=None, **w):
        super(group_norm, self).__init__()
        self.num_nodes = num_nodes
        self.num_groups = num_groups
        self.skip_weight = skip_weight
        self.dim_hidden = dim_hidden

        self.bn = torch.nn.BatchNorm1d(dim_hidden * self.num_groups * self.num_nodes, momentum=0.3)
        self.group_func = torch.nn.Linear(dim_hidden, self.num_groups, bias=True)

    def forward(self, x):
        if self.num_groups == 1:
            x_temp = self.bn(x)
        else:
            score_cluster = F.softmax(self.group_func(x), dim=2)
            x_temp = torch.cat([score_cluster[:, :, group].unsqueeze(dim=2) * x for group in range(self.num_groups)], dim=2)
            # batch, number_nodes, num_groups * dim_hidden
            x_temp = self.bn(x_temp.view(-1, self.num_nodes * self.num_groups * self.dim_hidden))
            x_temp = x_temp.view(-1, self.num_nodes, self.num_groups, self.dim_hidden).sum(dim=2)

        x = x + x_temp * self.skip_weight

        return x


def node_norm(x, p=2):
    """
    :param x: [batch, n_nodes, features]
    :return:
    """
    std_x = torch.std(x, dim=2, keepdim=True)
    x = x / (std_x ** (1 / p) + 1e-5)

    return x


def one_zero_normalization(x):
    if x.dim() == 4:
        dim_0 = x.size(0)
        dim_1 = x.size(1)
        dim_2 = x.size(2)
        dim_3 = x.size(3)

        x = x.view(dim_0, -1)
        x = x - x.min(dim=1, keepdim=True)[0]
        x = x / x.max(dim=1, keepdim=True)[0]
        x = x.view(-1, dim_1, dim_2, dim_3)

    elif x.dim() == 3:
        dim_0 = x.size(0)
        dim_1 = x.size(1)
        dim_2 = x.size(2)

        x = x.view(dim_0, -1)
        x = x - x.min(dim=1, keepdim=True)[0]
        x = x / x.max(dim=1, keepdim=True)[0]
        x = x.view(-1, dim_1, dim_2)

    return x


def mean_std_normalization(x):
    if x.dim() == 4:
        dim_0 = x.size(0)
        dim_1 = x.size(1)
        dim_2 = x.size(2)
        dim_3 = x.size(3)

        x = x.view(dim_0, -1)
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-12)
        x = x.view(-1, dim_1, dim_2, dim_3)

    elif x.dim() == 3:
        dim_0 = x.size(0)
        dim_1 = x.size(1)
        dim_2 = x.size(2)

        x = x.view(dim_0, -1)
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-12)
        x = x.view(-1, dim_1, dim_2)

    return x


def vec_l2_norm(x):
    """
    :param x: [Batch_size, num_feature]
    :return vector after L2 Normalization: [Batch_size, num_feature]
    """
    # x: [Batch_size, num_feature]
    if x.dim() == 2:
        norm = x.norm(p=2, dim=1, keepdim=True) + 1e-8

    # x: [Batch_size, num_node, num_feature]
    elif x.dim() == 3:
        norm = x.norm(p=2, dim=2, keepdim=True) + 1e-8

    l2_norm = x.div(norm)

    return l2_norm


def bilinear_pool(feature_1, feature_2):
    """
    :param feature_1: [Batch_size, num_feature]
    :param feature_2: [Batch_size, num_feature]
    :return bilinear pooling vector: [Batch_size, num_feature * num_feature]
    """
    num_feature = feature_1.size()[1]
    feature_1 = feature_1.unsqueeze(dim=1)  # [Batch_size, 1, num_feature]
    feature_2 = feature_2.unsqueeze(dim=1)  # [Batch_size, 1, num_feature]

    # [Batch_size, num_feature, 1] X [Batch_size, 1, num_feature] -> [Batch_size, num_feature, num_feature]
    xi = torch.bmm(torch.transpose(feature_1, 1, 2), feature_2)
    x = xi.view([-1, num_feature * num_feature])  # [Batch_size, num_feature * num_feature]
    y = torch.mul(torch.sign(x), torch.sqrt(torch.abs(x)))  # [Batch_size, num_feature * num_feature]
    z = vec_l2_norm(y)  # [Batch_size, num_feature * num_feature]

    return z


def gaussian_prior(mean, scale):
    noise = torch.randn(mean.size()).cuda()
    mos_pred = mean + noise * scale

    return mos_pred


def mkdirs(path):
    os.makedirs(path, exist_ok=True)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def image_show(path):
    image = mpimg.imread(path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def image_tensor_show(image_tensor):
    for i in range(np.shape(image_tensor)[0]):
        temp = image_tensor[i]
        temp = np.squeeze(temp, axis=0)
        temp = np.transpose(temp, (1, 2, 0))
        plt.imshow(temp)
        plt.axis('off')
        plt.show()


class SimpleProgressBar:
    def __init__(self, total_len, pat='#', show_step=False, print_freq=1):
        self.len = total_len
        self.pat = pat
        self.show_step = show_step
        self.print_freq = print_freq
        self.out_stream = stdout

    def show(self, cur, desc):
        bar_len, _ = shutil.get_terminal_size()
        # The tab between desc and the progress bar should be counted.
        # And the '|'s on both ends be counted, too
        bar_len = bar_len - self.len_with_tabs(desc + '\t') - 2
        bar_len = int(bar_len * 0.8)
        cur_pos = int(((cur + 1) / self.len) * bar_len)
        cur_bar = '|' + self.pat * cur_pos + ' ' * (bar_len - cur_pos) + '|'

        disp_str = "{0}\t{1}".format(desc, cur_bar)

        # Clean
        self.write('\033[K')

        if self.show_step and (cur % self.print_freq) == 0:
            self.write(disp_str, new_line=True)
            return

        if (cur + 1) < self.len:
            self.write(disp_str)
        else:
            self.write(disp_str, new_line=True)

        self.out_stream.flush()

    @staticmethod
    def len_with_tabs(s):
        return len(s.expandtabs())

    def write(self, content, new_line=False):
        end = '\n' if new_line else '\r'
        self.out_stream.write(content + end)
