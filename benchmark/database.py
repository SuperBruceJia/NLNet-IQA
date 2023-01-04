# -*- coding: utf-8 -*-

def database(benchmark, path):
    """
    Database Info
    :param benchmark: Database Name
    :param path: Main Path
    :return data_info: Database Information (image <-> MOS, and others)
    :return data_dir: Database Path
    """
    if benchmark == 'TID2013':
        data_dir = path + 'tid2013/distorted_images/'
        data_info = './benchmark/TID2013fullinfo.mat'
    elif benchmark == 'CSIQ':
        data_dir = path + 'CSIQ/'
        data_info = './benchmark/CSIQfullinfo.mat'
    elif benchmark == 'KADID':
        data_dir = path + 'kadid10k/images/'
        data_info = './benchmark/KADID-10K.mat'
    elif benchmark == 'LIVE':
        data_dir = path + 'live/'
        data_info = './benchmark/LIVEfullinfo.mat'
    else:
        data_dir = None
        data_info = None

    return data_info, data_dir


def distortion_type(database):
    """
    Distortion type info
    :param database: Database Name
    :return num_type: Number of Distortion Types
    """
    if database == 'LIVE':
        num_type = 5
    elif database == 'CSIQ':
        num_type = 6
    elif database == 'TID2013':
        num_type = 24
    elif database == 'KADID':
        num_type = 25
    else:
        num_type = None

    return num_type
