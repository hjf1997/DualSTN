"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.
It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import torch.utils.data as data
from abc import ABC, abstractmethod
import numpy as np
import os
import time
from functools import wraps
import torch
import random
import copy

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.
    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

    @staticmethod
    def split_dataset(A, X, missing_index, n_u, test_nodes_path):

        split_line1 = int(X.shape[1] * 0.7)
        if len(X.shape) == 3:
            training_set = X[:, :split_line1].transpose((1, 0, 2))
            training_missing_index = missing_index[:, :split_line1].transpose((1, 0, 2))
            test_set = X[:, split_line1:].transpose((1, 0, 2))  # split the training and test period
            test_missing_index = missing_index[:, split_line1:].transpose((1, 0, 2))  # split the training and test period
        else:
            training_set = X[:, :split_line1].transpose()
            training_missing_index = missing_index[:, :split_line1].transpose()
            test_set = X[:, split_line1:].transpose()  # split the training and test period
            test_missing_index = missing_index[:, split_line1:].transpose()

        if os.path.isfile(test_nodes_path):
            test_nodes = np.load(test_nodes_path)
            test_nodes = set(test_nodes)
        else:
            print('No testing nodes. Randomly divide nodes for testing!')
            rand = np.random.RandomState(0)  # Fixed random output
            test_nodes = rand.choice(list(range(0, X.shape[0])), n_u, replace=False)
            np.save(test_nodes_path, test_nodes)
            test_nodes = set(test_nodes)

        full_nodes = set(range(0, X.shape[0]))
        training_nodes = full_nodes - test_nodes

        training_set = training_set[:, list(training_nodes)]  # get the training data in the sample time period
        training_missing_index = training_missing_index[:, list(training_nodes)]
        A_training = A[:, list(training_nodes)][list(training_nodes), :]  # get the observed adjacent matrix from the full adjacent matrix,
        # the adjacent matrix are based on pairwise distance,
        # so we need not construct it for each batch, we just use index to find the dynamic adjacent matrix
        return training_set, training_missing_index, test_set, test_missing_index, training_nodes, test_nodes, A_training

    def collate_fn(self, batch):
        """
        Collate function to be used when wrapping a generator.
        """
        ground_truth = torch.cat([d['gt'].unsqueeze(0) for d in batch], dim=0)
        missing_index = torch.cat([d['missing_index'].unsqueeze(0) for d in batch], dim=0)
        test_nodes_index = batch[0]['test_nodes_index']
        A = batch[0]['adj']  # list [ , ]
        dist = batch[0]['dist']

        if not test_nodes_index:  # training type:set
            test_nodes_index = random.sample(range(0, ground_truth.shape[-1]-5), self.opt.num_train_target)
        test_nodes_index = sorted(test_nodes_index)

        sample = copy.deepcopy(ground_truth)
        sample[:, :, test_nodes_index] = 0
        test_nodes_map = torch.zeros_like(sample)
        test_nodes_map[:, :, test_nodes_index] = 1
        test_nodes_map = (test_nodes_map * (1 - missing_index)).int()

        # preinterpolate
        sample = self.preinterpolate(dist, sample, test_nodes_index)
        return {
            'gt': ground_truth,  # [batch, time, num_n]
            'sample': sample,
            'test_nodes_index': test_nodes_map,
            'missing_index': missing_index,
            'adj': A
        }

    def preinterpolate(self, adj, samples, test_nodes, K=5):
        """
        As we found that initialize values of nodes to be inferred to 0 will degrade the performance of models.
        Here, we pre-interpolate these nodes
        :param adj: [num_nodes, num_nodes]
        :param samples: [num_samples, num_nodes, (num_features)]
        :param test_nodes: [num_test_nodes]
        :param K: KNN
        :return:
        """
        if K == 0:
            # no pre-interpolation
            return samples
        if K == -1:
            # use all neighbors
            K = samples.shape[1]

        if samples.shape[1] != adj.shape[0]:
            batch, time = samples.shape[:2]
            samples = samples.reshape(-1, *samples.shape[2:])
        else:
            batch, time = samples.shape[0], 1

        all_nodes_index = set(range(0, samples.shape[1]))
        training_nodes_list = list(all_nodes_index - set(test_nodes))
        test_nodes_list = list(test_nodes)

        for index in range(samples.shape[0]):
            sample = samples[index, training_nodes_list]  # [num_known_nodes, num_features]
            adj_graph = adj[test_nodes_list][:, training_nodes_list]
            adj_graph[adj_graph == 0] = float('inf')
            sample = np.expand_dims(sample, axis=0)
            adj_column_index = np.expand_dims(np.arange(adj_graph.shape[0], dtype=np.int32), axis=1)
            graph_column_index = np.expand_dims(np.zeros(adj_graph.shape[0], dtype=np.int32), axis=1)
            topk_index = np.argpartition(adj_graph, K, axis=1)[:, 0:K]
            adj_graph = adj_graph[adj_column_index, topk_index]
            adj_graph = 1 / (adj_graph + 1e-6)  # inverse distance weighted
            s = sample[graph_column_index, topk_index]
            knn_graph = np.sum(s * adj_graph, axis=1) / (np.sum(adj_graph, axis=1) + 1e-6)
            samples[index, test_nodes_list] = torch.from_numpy(knn_graph).float()

        if time != 1:
            samples = samples.reshape(batch, time, *samples.shape[1:])
        return samples

def timefn(fn):
    """计算性能的修饰器"""
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1: .5f} s")
        return result
    return measure_time


# @nb.jit(nopython=False, parallel=True)
@timefn
def knn(adj, graphs, unknown_list, K, filter_zero=False):
    """
    knn for each graph
    :param adj: adjacent matrix [num_nodes, num_nodes]
    :param graphs: graphs [num_graphs，num_nodes, num_features]
    :param unknown_set: list indicator of known nodes
    :param K: int K nearest neighbours
    :return:
    """
    full_list = set(range(0, graphs.shape[1]))
    known_list = list(full_list - set(unknown_list))

    for graph_i in range(graphs.shape[0]):
        graph = graphs[graph_i, known_list]  # [num_known_nodes, num_features]
        adj_graph = adj[unknown_list][:, known_list]
        adj_graph[adj_graph == 0] = float('inf')
        graph = np.expand_dims(graph, axis=0)
        adj_column_index = np.expand_dims(np.arange(adj_graph.shape[0], dtype=np.int32), axis=1)
        graph_column_index = np.expand_dims(np.zeros(adj_graph.shape[0], dtype=np.int32), axis=1)
        topk_index = np.argpartition(adj_graph, K, axis=1)[:, 0:K]
        adj_graph = np.expand_dims(adj_graph[adj_column_index, topk_index], axis=2)  # broadcasting
        knn_graph = np.mean(graph[graph_column_index, topk_index, :] * adj_graph, axis=1) / np.sum(adj_graph, axis=1)
        graphs[graph_i, unknown_list] = knn_graph

    return graphs

if __name__ == '__main__':
    knn(np.random.rand(250, 250), np.random.rand(16*24, 250, 3), [i for i in range(50)], 6)