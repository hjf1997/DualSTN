"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
import os
from utils.operation import calculate_random_walk_matrix, calculate_normalized_laplacian, calculate_sum_norm


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)
    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array
    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def _rmse_with_missing(y, label, valid_index):
    """
    Args:
        y: Tensor [(batch), time, num_m, dy]
        label: Tensor
        missing_index: [(batch), time, num_m, 1]
    Returns:
        mse: float
    """
    valid_count = torch.sum(valid_index)

    rmse = torch.sqrt((((y - label) ** 2) * valid_index).sum() / (valid_count + 1e-7))

    return rmse.item()


def _mae_with_missing(y, label, valid_index):
    """
    Args:
        y: Tensor [(batch), time, num_m, dy]
        label: Tensor
        valid_index: [(batch), time, num_m, 1]
    Returns:
        mae: float
    """
    valid_count = torch.sum(valid_index)

    mas = torch.abs((y-label) * valid_index).sum() / valid_count
    return mas.item()

def _mape_with_missing(y, label, valid_index):
    """
    Args:
        y: Tensor [(batch), time, num_m, dy]
        label: Tensor
        valid_index: [(batch), time, num_m, 1]
    Returns:
        mae: float
    """
    valid_index = valid_index * (torch.abs(label) > 0.0001)
    valid_count = torch.sum(valid_index)

    mape = torch.abs((y-label) / (label+1e-6) * valid_index).sum() / valid_count
    return mape.item()

def _r2_with_missing(y, label, valid_index):
    """
    Args:
        y: Tensor [(batch), time, num_m]
        label: Tensor
        valid_index: [(batch), time, num_m]
    Returns:
        mae: float
    """
    mean = torch.sum((y - label) * valid_index) / torch.sum(valid_index)
    r2 = 1 - torch.sum((y-label) ** 2 * valid_index) / torch.sum((y-mean) ** 2 * valid_index)
    return r2.item()

def norm_adj(A):
    # random walk matrix
    if type(A) == list:
        return [torch.from_numpy(calculate_normalized_laplacian(adj)).float() for adj in A]
    else:
        return  torch.from_numpy(calculate_normalized_laplacian(A)).float()