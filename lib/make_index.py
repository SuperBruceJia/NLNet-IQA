# -*- coding: utf-8 -*-

import random
import numpy as np
from PIL import Image


def make_index(args, index):
    """
    Split the reference images indexes into training, validation, and testing
    :param args: The configurations
    :param index: The reference images indexes
    :return: train_index, val_index, test_index
    """
    # Get reference index and its length
    reference_index = np.sort(index)
    num_ref = len(reference_index)

    # Shuffle the indexes by the SEED args.exp_id
    random_seed = args.exp_id
    random.seed(random_seed)
    random.shuffle(reference_index)

    # Split the indexes by 60% training, 20% validation, and 20% testing
    train_index = reference_index[:int(num_ref * 0.6)]
    val_index = reference_index[int(num_ref * 0.6):int(num_ref * 0.8)]
    test_index = reference_index[int(num_ref * 0.8):]

    return train_index, val_index, test_index


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
