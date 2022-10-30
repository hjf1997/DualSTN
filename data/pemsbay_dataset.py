# implemented by p0werHu
# Time: 20/07/2021
import numpy as np
import torch
import random
from data import BaseDataset
import os
import pandas as pd


class PeMSBayDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.set_defaults(n_test=160, num_train_target=80)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.t_len = opt.t_len

        assert os.path.isfile('dataset/pemsbay/pems-bay.h5')
        assert os.path.isfile('dataset/pemsbay/distances_bay_2017.csv')
        df = pd.read_hdf('dataset/pemsbay/pems-bay.h5')
        transfer_set = df.values.T
        distance_df = pd.read_csv('dataset/pemsbay/distances_bay_2017.csv', dtype={'from': 'str', 'to': 'str'})

        dist_mx = np.zeros((325, 325), dtype=np.float32)

        dist_mx[:] = np.inf
        z_mask = np.eye(325, dtype=np.float32)
        dist_mx[z_mask == 1] = 0
        sensor_ids = df.columns.values.tolist()
        sensor_id_to_ind = {}

        for i, sensor_id in enumerate(sensor_ids):
            sensor_id_to_ind[sensor_id] = i

        for row in distance_df.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                continue
            dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

        distances = dist_mx[~np.isinf(dist_mx)].flatten()
        std = distances.std()
        adj_mx = np.exp(-0.5 * np.square(dist_mx / std))

        A = adj_mx
        test_nodes_path = 'dataset/pemsbay/test_nodes.npy'

        self.dist = dist_mx
        self.A = adj_mx
        X = transfer_set
        missing_index = np.zeros(X.shape)
        missing_index[X == 0] = 1

        self.training_set, self.training_missing_index, \
        self.test_set, self.test_missing_index, \
        self.training_nodes, self.test_nodes, \
        self.A_training = self.split_dataset(A, X, missing_index, opt.n_test, test_nodes_path)
        self.dist_training = dist_mx[:, list(self.training_nodes)][list(self.training_nodes), :]

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
            adj = [torch.from_numpy(self.A_training).float(), torch.from_numpy(self.A_training.T).float()]
            dist = self.dist_training
        else:
            gt = self.test_set[index: index + self.t_len, :]
            missing_index = self.test_missing_index[index: index + self.t_len, :]  # index for missing signals
            test_nodes_index = self.test_nodes
            adj = [torch.from_numpy(self.A).float(), torch.from_numpy(self.A.T).float()]
            dist = self.dist

        gt = torch.from_numpy(gt).float()
        missing_index = torch.from_numpy(missing_index).int()

        return {'gt': gt,
                'dist': dist,
                'missing_index': missing_index,
                'test_nodes_index': test_nodes_index,
                'adj': adj}
