# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

from data_process.get_data import IQADataset


def data_loader(args):
    """
    Prepare the train-val-test data
    :param args: related arguments
    :return: train_loader, test_loader
    """
    train_dataset = IQADataset(args=args, status='train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=True)
    
    val_dataset = IQADataset(args=args, status='val')
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True)

    test_dataset = IQADataset(args=args, status='test')
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    return train_loader, val_loader, test_loader
