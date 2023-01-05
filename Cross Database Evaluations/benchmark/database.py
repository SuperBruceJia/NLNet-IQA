#!/usr/bin/env python
# -*- coding: utf-8 -*-


def cross_database(train, test, path):
    # Training Database
    if train == 'TID2013':
        train_dir = path + 'tid2013/'
        train_info = './benchmark/TID2013fullinfo.mat'
        train_im_dir = train_dir + 'distorted_images/'
        train_ref_dir = train_dir + 'reference_images/'
    elif train == 'LIVE':
        train_dir = path + 'live/'
        train_info = './benchmark/LIVEfullinfo.mat'
        train_im_dir = train_dir
        train_ref_dir = train_dir + 'refimgs/'
    elif train == 'CSIQ':
        train_dir = path + 'CSIQ/'
        train_info = './benchmark/CSIQfullinfo.mat'
        train_im_dir = train_dir
        train_ref_dir = train_dir + 'refimgs/'
    elif train == 'KADID':
        train_dir = path + 'kadid10k/images/'
        train_info = './benchmark/KADID-10K.mat'
        train_im_dir = train_dir
        train_ref_dir = train_dir + 'refimgs/'
    else:
        train_dir = None
        train_info = None
        train_im_dir = None
        train_ref_dir = None

    # Testing Database
    if test == 'TID2013':
        test_dir = path + 'tid2013/'
        test_info = './benchmark/TID2013fullinfo.mat'
        test_im_dir = test_dir + 'distorted_images/'
        test_ref_dir = test_dir + 'reference_images/'
    elif test == 'LIVE':
        test_dir = path + 'live/'
        test_info = './benchmark/LIVEfullinfo.mat'
        test_im_dir = test_dir
        test_ref_dir = test_dir + 'refimgs/'
    elif test == 'CSIQ':
        test_dir = path + 'CSIQ/'
        test_info = './benchmark/CSIQfullinfo.mat'
        test_im_dir = test_dir
        test_ref_dir = test_dir + 'refimgs/'
    elif test == 'KADID':
        test_dir = path + 'kadid10k/images/'
        test_info = './benchmark/KADID-10K.mat'
        test_im_dir = test_dir
        test_ref_dir = test_dir + 'refimgs/'
    elif test == 'LIVEC':
        test_dir = path + 'LIVEC/Images/'
        test_info = './benchmark/CLIVEinfo.mat'
        test_im_dir = test_dir
        test_ref_dir = test_dir + 'refimgs/'
    elif test == 'KonIQ':
        test_dir = path + 'koniq10k/512x384/'
        test_info = './benchmark/KonIQ-10k.mat'
        test_im_dir = test_dir
        test_ref_dir = test_dir + 'refimgs/'
    elif test == 'BID':
        test_dir = path + 'BID/BID/ImageDatabase/'
        test_info = './benchmark/BIDinfo.mat'
        test_im_dir = test_dir
        test_ref_dir = test_dir + 'refimgs/'
    elif test == 'SIQAD':
        test_dir = path + 'SIQAD/DistortedImages/'
        test_info = './benchmark/SIQADinfo.mat'
        test_im_dir = test_dir
        test_ref_dir = test_dir + 'refimgs/'
    elif test == 'SCID':
        test_dir = path + 'SCID/DistortedSCIs/'
        test_info = './benchmark/SCIDinfo.mat'
        test_im_dir = test_dir
        test_ref_dir = test_dir + 'refimgs/'
    else:
        test_dir = None
        test_info = None
        test_im_dir = None
        test_ref_dir = None

    return train_info, train_im_dir, train_ref_dir, test_info, test_im_dir, test_ref_dir


def distortion_type(database):
    """
    Number of distortion types
    """
    if database == 'LIVE':
        num_type = 5
    elif database == 'CSIQ':
        num_type = 6
    elif database == 'TID2008':
        num_type = 17
    elif database == 'TID2013':
        num_type = 24
    elif database == 'KADID':
        num_type = 25
    else:
        num_type = None

    return num_type
