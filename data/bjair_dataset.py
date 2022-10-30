# implemented by Fan Zhencheng
# Time: 14/06/2021
from data.base_dataset import BaseDataset
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import networkx as nx
import numpy as np
import torch


class BJAirDataset(BaseDataset):
    """
    Note that the beijing air quality dataset contains a lot of missing values, we need to handle this explicitly.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.set_defaults(in_dim=1, n_test=8, num_train_target=8)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        """
        Get adjacency matrix and [stations_num, time_num, 6_air_quality]
        """
        self.t_len = opt.t_len
        A, original_A, X, missing_index, bj_time = self.get_adjacency_matrix_and_data('dataset/bjair/Beijing_AirQuality_Stations_cn_for_read.xlsx',
                                                     ['dataset/bjair/beijing_201802_201803_aq.csv', 'dataset/bjair/beijing_17_18_aq.csv'])
        X = X.astype(np.float32)
        original_A = original_A.astype(np.float32)

        test_nodes_path = 'dataset/bjair/test_nodes.npy'

        self.dist = original_A
        self.A = A
        self.training_set, self.training_missing_index,\
        self.test_set, self.test_missing_index,\
        self.training_nodes, self.test_nodes, \
        self.A_training = self.split_dataset(A, X, missing_index, opt.n_test, test_nodes_path)
        self.dist_training = original_A[:, list(self.training_nodes)][list(self.training_nodes), :]
        self.test_time = bj_time[int(len(bj_time) * 0.7):]  # Only used for attention scores visualization
        # print(self.test_set.shape)

        # data normalization
        self.opt.mean = self.training_set.mean()
        self.opt.std = self.training_set.std()
        self.training_set = (self.training_set - self.opt.mean) / self.opt.std
        self.test_set = (self.test_set - self.opt.mean) / self.opt.std

    def __len__(self):
        """Return the total number of time steps in the dataset."""
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

        return_dir = {
            'gt': gt,
            'dist': dist,
            'missing_index': missing_index,
            'test_nodes_index': test_nodes_index,
            'adj': adj,
        }
        # if self.opt.phase == 'test':
        #     return_dir['time'] = torch.tensor([self.test_time[index + self.t_len]])  # only used for testing!!!
        # else:
        #     return_dir['time'] = torch.tensor([self.test_time[0]])

        return return_dir

    @property
    def get_test_data_time(self):
        return self.test_time

    def get_adjacency_matrix_and_data(self, stations_file_path, beijing_aq_paths):
        """
        Ordered by 'stationId' and get adjacency matrix
        35 stations in total
        Get aq data ordered by ['stationId', 'time']
        beijing_201802_201803_aq.csv
        beijing_17_18_aq.csv
        PM2.5	PM10	NO2	CO	O3	SO2
        return adjacency matrix and [stations_num, time_num, 6_air_quality]
        """
        stations_data = pd.read_excel(stations_file_path).to_numpy()
        stations_data = stations_data[np.lexsort(stations_data[:, ::-1].T)]
        stations_list = stations_data[:, 0]
        G = nx.Graph()
        G.add_nodes_from(stations_list)
        for i in range(0, len(stations_list)):
            for j in range(i + 1, len(stations_list)):
                G.add_edge(stations_list[i], stations_list[j],
                           weight=self.haversine(stations_data[i][1], stations_data[i][2], stations_data[j][1],
                                                 stations_data[j][2]))
        A = nx.adjacency_matrix(G).todense()

        # Gaussian kernel
        original_A = A.A
        A = np.exp(- 0.5 * (original_A / np.std(original_A, axis=1, keepdims=True)) ** 2)

        # here we use interpolate
        aq_data = []
        for path in beijing_aq_paths:
            aq_data.append(pd.read_csv(path))
        # interpolate missing values
        beijing_aq = pd.concat(aq_data).sort_values(
            by=['stationId', 'utc_time']).drop(columns=['stationId', 'utc_time'])
        missing_index = beijing_aq.isna().to_numpy().reshape((len(stations_list), -1, 6))
        beijing_aq = beijing_aq.interpolate('linear', axis=0).ffill().bfill().fillna(0)
        beijing_aq = beijing_aq.to_numpy().reshape((len(stations_list), -1, 6))
        beijing_time = pd.concat(aq_data).sort_values(by=['stationId', 'utc_time'])['utc_time'].unique().tolist()

        # PM 2.5
        beijing_aq = beijing_aq[..., 0]
        missing_index = missing_index[..., 0]

        return A, original_A, beijing_aq, missing_index, beijing_time

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
