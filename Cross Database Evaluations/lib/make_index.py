#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image


def make_data_index(args, index):
    '''
    :param args:
    :param index:
    :return:
    '''
    # K is for K-fold cross-validation
    # k is The k-th fold used for test
    K = args.K_fold  # Here, we don't use this because we use 10-fold cross-validation in default
    k = args.k_test
    # print('The index is ', index, 'Length of Index', len(index))

    # Here we assume to use 10-fold Cross-validation, i.e., K = 10
    # Approximately 60% Training set, 20% Validation set, 20% Testing set
    num_test_ref = int(len(index) * 0.2)
    num_val_ref = int(len(index) * 0.2)
    num_train_ref = len(index) - num_test_ref - num_val_ref

    # Assume k = 1 : 10 for 10-fold Cross-validation
    threshold = int(len(index) / num_test_ref)

    if k < threshold:
        testindex = index[(k - 1) * num_test_ref: k * num_test_ref]
        valindex = index[k * num_val_ref: (k + 1) * num_val_ref]

    elif k == threshold:
        testindex = index[(k - 1) * num_test_ref: k * num_test_ref]
        # Check if the index num of validation set is less than num_val_ref
        valindex = index[k * num_val_ref: (k + 1) * num_val_ref]
        if len(valindex) < num_val_ref:
            valindex = valindex.tolist()
            for i in range(0, num_val_ref - len(valindex)):
                valindex.append(index[i])

    elif k == threshold + 1:
        testindex = index[k * num_test_ref: (k + 1) * num_test_ref]
        if len(testindex) < num_test_ref:
            testindex = testindex.tolist()
            for i in range(0, num_test_ref - len(testindex)):
                testindex.append(index[i])

        k -= threshold
        valindex = index[(k + 2) * num_val_ref: (k + 3) * num_val_ref]

    else:
        k -= threshold
        testindex = index[k * num_test_ref: (k + 1) * num_test_ref]
        if len(testindex) < num_test_ref:
            testindex = testindex.tolist()
            for i in range(0, num_test_ref - len(testindex)):
                testindex.append(index[i + num_test_ref])

        valindex = index[(k + 2) * num_val_ref: (k + 3) * num_val_ref]
        if len(valindex) < num_val_ref:
            valindex = valindex.tolist()
            for i in range(0, num_val_ref - len(valindex)):
                valindex.append(index[i])

    return valindex, testindex


def default_loader(path, channel=3):
    """
    :param path: image path
    :param channel: # image channel
    :return: image
    """
    if channel == 1:
        return Image.open(path).convert('L')
    else:
        assert (channel == 3)
        return Image.open(path).convert('RGB')
