# Implemented by p0werHu
# Time 11/06/2021
# Adopted from https://github.com/Kaimaoge/IGNNK/blob/master/utils.py

import pandas as pd
from scipy.io import loadmat

from data.base_dataset import BaseDataset
import os
import numpy as np
import random
import torch
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm


class NrelDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.set_defaults(n_test=68,num_train_target=32)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.t_len = opt.t_len

        X = np.load('dataset/nrel/nerl_X.npy')
        files_info = pd.read_pickle('dataset/nrel/nerl_file_infos.pkl')
        dist = loadmat('dataset/nrel/nrel_dist_mx_lonlat.mat')['nrel_dist_mx_lonlat']
        dist = dist / 1000  # convert to km

        self.dist = dist
        A = np.exp(- 0.5 * (dist / np.std(dist, axis=1, keepdims=True)) ** 2)
        X = X.astype(np.float32)
        missing_index = np.zeros(X.shape) # no missing values

        #For Nrel, We only use 7:00am to 7:00pm as the target data, because otherwise the 0-values of periods without sunshine will greatly influence the results
        time_used_base = np.arange(84, 228)
        time_used = np.array([])
        for i in range(365):
            time_used = np.concatenate((time_used,time_used_base + 24*12* i))
        X = X[:, time_used.astype(np.int)]
        capacities = np.array(files_info['capacity'])
        self.capacities = capacities.astype('float32')

        test_nodes_path = 'dataset/nrel/test_nodes.npy'
        self.A = A
        self.training_set, self.training_missing_index, \
        self.test_set, self.test_missing_index, \
        self.training_nodes, self.test_nodes, \
        self.A_training = self.split_dataset(A, X, missing_index, opt.n_test, test_nodes_path)
        self.dist_training = dist[:, list(self.training_nodes)][list(self.training_nodes), :]
        # data normalization 0-1 normalization
        self.opt.mean = 0
        self.opt.std = torch.from_numpy(np.expand_dims(self.capacities, axis=-1))
        self.training_set = self.training_set / self.capacities[list(self.training_nodes)]
        self.test_set = self.test_set / self.capacities

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

    def generate_nerl_data(self):
        # %% Obtain all the file names
        filepath = 'dataset/nrel/al-pv-2006'
        files = os.listdir(filepath)

        # %% Begin parse the file names and store them in a pandas Dataframe
        tp = []  # Type
        lat = []  # Latitude
        lng = []  # Longitude
        yr = []  # Year
        pv_tp = []  # PV_type
        cap = []  # Capacity MW
        time_itv = []  # Time interval
        file_names = []
        for _file in files:
            parse = _file.split('_')
            if parse[-2] == '5':
                tp.append(parse[0])
                lat.append(np.double(parse[1]))
                lng.append(np.double(parse[2]))
                yr.append(np.int(parse[3]))
                pv_tp.append(parse[4])
                cap.append(np.int(parse[5].split('MW')[0]))
                time_itv.append(parse[6])
                file_names.append(_file)
            else:
                pass

        files_info = pd.DataFrame(
            np.array([tp, lat, lng, yr, pv_tp, cap, time_itv, file_names]).T,
            columns=['type', 'latitude', 'longitude', 'year', 'pv_type', 'capacity', 'time_interval', 'file_name']
        )
        # %% Read the time series into a numpy 2-D array with 137x105120 size
        X = np.zeros((len(files_info), 365 * 24 * 12))
        for i in tqdm(range(files_info.shape[0]), desc='Processing Data'):
            f = filepath + '/' + files_info['file_name'].loc[i]
            d = pd.read_csv(f)
            assert d.shape[0] == 365 * 24 * 12, 'Data missing!'
            X[i, :] = d['Power(MW)']

        np.save('dataset/nrel/nerl_X.npy', X)
        files_info.to_pickle('dataset/nrel/nerl_file_infos.pkl')
        # %% Get the adjacency matrix based on the inverse of distance between two nodes
        A = np.zeros((files_info.shape[0], files_info.shape[0]))

        for i in range(files_info.shape[0]):
            for j in range(i + 1, files_info.shape[0]):
                lng1 = lng[i]
                lng2 = lng[j]
                lat1 = lat[i]
                lat2 = lat[j]
                d = self.haversine(lng1, lat1, lng2, lat2)
                A[i, j] = d

        A = A / 7500  # distance / 7.5 km
        A += A.T + np.diag(A.diagonal())
        A = np.exp(-A)
        np.save('dataset/nrel/nerl_A.npy', A)

    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        r = 6371
        return c * r
