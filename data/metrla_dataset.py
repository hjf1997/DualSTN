# Implemented by p0werHu
# Time 13/06/2021
# Adopted from https://github.com/Kaimaoge/IGNNK/blob/master/utils.py
from scipy.io import loadmat

from data.base_dataset import BaseDataset
import os
import numpy as np
import zipfile
import random
import torch


class MetrlaDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.set_defaults(n_test=138,num_train_target=30)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.t_len = opt.t_len

        if (not os.path.isfile("dataset/metr/adj_mat.npy")
                or not os.path.isfile("dataset/metr/node_values.npy")):
            with zipfile.ZipFile("dataset/metr/METR-LA.zip", 'r') as zip_ref:
                zip_ref.extractall("dataset/metr/")

        # A = np.load("dataset/metr/adj_mat.npy")
        dis = loadmat('dataset/metr/dist_metrla.mat')
        dis = dis['dis']
        # original_A = dist_mx  # Just for KNN
        A = np.exp(- 0.5 * (dis / np.std(dis, axis=1, keepdims=True)) ** 2)
        X = np.load("dataset/metr/node_values.npy").transpose((1, 2, 0))
        X = X.astype(np.float32)
        X = X[:, 0, :]
        X = X
        # following IGNNK, we regard 0 as missing value
        missing_index = np.zeros(X.shape)
        missing_index[X == 0] = 1

        test_nodes_path = 'dataset/metr/test_nodes.npy'

        self.dist = dis
        self.A = A
        self.training_set, self.training_missing_index, \
        self.test_set, self.test_missing_index, \
        self.training_nodes, self.test_nodes, \
        self.A_training = self.split_dataset(A, X, missing_index, opt.n_test, test_nodes_path)
        self.dist_training = dis[:, list(self.training_nodes)][list(self.training_nodes), :]

        # data normalization
        self.opt.mean = self.training_set.mean()
        self.opt.std = self.training_set.std()
        self.training_set = (self.training_set - self.opt.mean) / self.opt.std
        self.test_set = (self.test_set - self.opt.mean) / self.opt.std

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.opt.phase == 'train':
            length = self.training_set.shape[0] - self.t_len
        else:
            length = self.test_set.shape[0] - self.t_len
        return length

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        """
        if self.opt.phase == 'train':
            gt = self.training_set[index: index + self.t_len, :]
            missing_index = self.training_missing_index[index: index + self.t_len, :]  # index for missing signals
            test_nodes_index = None
            # adj = self.norm_adj(self.A_training)
            adj = [torch.from_numpy(self.A_training).float(), torch.from_numpy(self.A_training).float()]
            dist = self.dist_training
        else:
            gt = self.test_set[index: index + self.t_len, :]
            missing_index = self.test_missing_index[index: index + self.t_len, :]  # index for missing signals
            test_nodes_index = self.test_nodes
            # adj = self.norm_adj(self.A)
            adj = [torch.from_numpy(self.A).float(), torch.from_numpy(self.A).float()]
            dist = self.dist

        gt = torch.from_numpy(gt).float()
        missing_index = torch.from_numpy(missing_index).int()

        return {'gt': gt,
                'dist': dist,
                'missing_index': missing_index,
                'test_nodes_index': test_nodes_index,
                'adj': adj}
