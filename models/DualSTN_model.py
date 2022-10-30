# implemented by p0werHu
# 30/05/2021
from collections import OrderedDict
from torch.autograd import Variable
from utils.operation import *
from . import init_net, BaseModel
import torch.nn as nn
from utils.util import _mae_with_missing, _rmse_with_missing, _mape_with_missing, _r2_with_missing, norm_adj


class DualSTNModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        """
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        TGN_param = {
            'layer1': {'dims_TGN': [-1, 8], 'depth_TGN': 3},
            'layer2': {'dims_TGN': [8, 16], 'depth_TGN': 3},
            'layer3': {'dims_TGN': [16, 32], 'depth_TGN': 3},
        }
        hyperGCN_param = {'dims_hyper': [-1, 16], 'depth_GCN': 1}

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['short_MAE', 'long_MAE']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['LSJSTN']

        # define networks
        self.netLSJSTN = LSJSTN(opt.in_dim, hyperGCN_param, TGN_param,
                              opt.hidden_size, opt.dropout, opt.tanhalpha)
        self.netLSJSTN = init_net(self.netLSJSTN, opt.init_type, opt.init_gain, opt.gpu_ids)

        # define loss functions
        if self.isTrain:
            self.criterion = self.mse_loss
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(self.netLSJSTN.parameters(), lr=opt.lr, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer)

        # results cache
        self.results = {}

    def set_input(self, input):
        self.sample = input['sample'].to(self.device)  # [batch, time, num_n]
        self.sample = self.sample.unsqueeze(-1)  # [batch, time, num_n, 1]

        self.gt = input['gt'].to(self.device)[:, -1, :].unsqueeze(-1)  # [batch, num_n, 1]

        self.test_nodes_index = input['test_nodes_index'].to(self.device)[:, -1, :].unsqueeze(-1)  # [batch, num_n, 1]
        self.missing_index = input['missing_index'].to(self.device)[:, -1, :].unsqueeze(-1)  # [batch, num_n, 1]
        self.predefined_A = norm_adj([input['adj'][0], input['adj'][1]])
        self.predefined_A = (self.predefined_A[0].to(self.device) + self.predefined_A[1].to(self.device)) / 2
        # self.time = input['time']  # [batch]

    def mse_loss(self, y, label, valid_index):
        valid_count = torch.sum(valid_index)
        mse = (((y - label) ** 2) * valid_index).sum() / (valid_count + 1e-7)
        return mse

    def mae_loss(self, y, label, valid_index):
        valid_count = torch.sum(valid_index)
        mae = torch.sum(torch.abs(y - label) * valid_index) / (valid_count + 1e-7)
        return mae

    def forward(self):
        self.gat_outputs, self.gru_outputs, self.outputs, self.atten_scores = self.netLSJSTN(self.sample, self.predefined_A)

    def backward(self):
        self.loss_long_MAE = self.criterion(self.outputs, self.gt, 1 - self.missing_index)
        self.loss_short_MAE = self.criterion(self.gat_outputs, self.gt, 1 - self.missing_index)
        loss_mid_MAE = self.criterion(self.gru_outputs, self.gt, 1 - self.missing_index)

        loss_MAE = self.loss_long_MAE + self.loss_short_MAE + loss_mid_MAE
        loss_MAE.backward()

    def cache_results(self):
        if 'short_results' not in self.results.keys():
            self.results['short_results'] = []
        self.results['short_results'].append(self.inverse_norm(self.gat_outputs))
        if 'long_results' not in self.results.keys():
            self.results['long_results'] = []
        self.results['long_results'].append(self.inverse_norm(self.outputs))
        if 'ground_truth' not in self.results.keys():
            self.results['ground_truth'] = []
        self.results['ground_truth'].append(self.inverse_norm(self.gt))
        if 'test_nodes_index' not in self.results.keys():
            self.results['test_nodes_index'] = []
        self.results['test_nodes_index'].append(self.test_nodes_index)
        # if 'time' not in self.results.keys():
        #     self.results['time'] = []
        # self.results['time'].append(self.time)
        if 'attention' not in self.results.keys():
            self.results['attention'] = []
        self.results['attention'].append(self.atten_scores)
        
    def save_data(self):
        for key, values in self.results.items():
            if type(values) == list:
                self.results[key] = torch.cat(values, 0)
            self.results[key] = values.cpu().numpy()
        super().save_data(self.results)

    def compute_metrics(self):
        for key, values in self.results.items():
            self.results[key] = torch.cat(values, dim=0)
        self.metrics['short_RMSE'] = _rmse_with_missing(self.results['short_results'], self.results['ground_truth'], self.results['test_nodes_index'])
        self.metrics['short_MAE'] = _mae_with_missing(self.results['short_results'], self.results['ground_truth'], self.results['test_nodes_index'])
        self.metrics['short_MAPE'] = _mape_with_missing(self.results['short_results'], self.results['ground_truth'], self.results['test_nodes_index'])
        self.metrics['short_R2'] = _r2_with_missing(self.results['short_results'], self.results['ground_truth'], self.results['test_nodes_index'])
        self.metrics['RMSE'] = _rmse_with_missing(self.results['long_results'], self.results['ground_truth'], self.results['test_nodes_index'])
        self.metrics['MAE'] = _mae_with_missing(self.results['long_results'], self.results['ground_truth'], self.results['test_nodes_index'])
        self.metrics['MAPE'] = _mape_with_missing(self.results['long_results'], self.results['ground_truth'], self.results['test_nodes_index'])
        self.metrics['R2'] = _r2_with_missing(self.results['long_results'], self.results['ground_truth'], self.results['test_nodes_index'])

    def clear_cache(self):
        self.results = {}

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netLSJSTN, True)
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


class LSJSTN(nn.Module):

    def __init__(self,
                 in_dim,
                 hyperGCN_param,
                 TGN_param,
                 hidden_size,
                 dropout=0.2,
                 tanhalpha=2,
                 list_weight=[0.05, 0.95, 0.95]):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.alpha = tanhalpha

        # Section: TGN for short-term learning
        self.TGNs = nn.ModuleList()
        for layer_name, layer_param in TGN_param.items():
            if layer_param['dims_TGN'][0] == -1:
                layer_param['dims_TGN'][0] = in_dim
            TGN_block = nn.ModuleDict({
                'b_tga': TAttn(layer_param['dims_TGN'][0]),
                'b_tgn1': GCN(layer_param['dims_TGN'], layer_param['depth_TGN'], dropout, *list_weight, 'TGN'),
            })
            self.TGNs.append(TGN_block)
        self.TGN_layers = len(self.TGNs)

        # Section: Hyper Adaptive graph generation
        # Subsection: define hyperparameter for adaptive graph generation
        dims_hyper = hyperGCN_param['dims_hyper']
        dims_hyper[0] = hidden_size
        gcn_depth = hyperGCN_param['depth_GCN']

        # Subsection: GCN and node embedding for adaptive graph generation
        self.GCN_agg1 = GCN(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.GCN_agg2 = GCN(dims_hyper, gcn_depth, dropout, *list_weight, 'hyper')
        self.source_nodeemb = nn.Linear(self.in_dim, dims_hyper[-1])
        self.target_nodeemb = nn.Linear(self.in_dim, dims_hyper[-1])

        # Section: Long-term recurrent graph GRU learning
        dims = [self.in_dim + self.hidden_size, self.hidden_size]
        self.gz1_de = GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gr1_de = GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')
        self.gc1_de = GCN(dims, gcn_depth, dropout, *list_weight, 'RNN')

        # Section: Final output linear transformation
        self.fc_final_short = nn.Linear(layer_param['dims_TGN'][-1], self.in_dim)
        self.fc_final_long = nn.Linear(self.hidden_size, self.in_dim)
        self.fc_final_mix = nn.Linear(layer_param['dims_TGN'][-1] + self.hidden_size, self.in_dim)

    def forward(self, sample, predefined_A):
        """
        Kriging for one iteration
        :param sample: graphs [batch, num_timesteps，num_nodes, num_features]
        :param predefined_A: list, len(2)
        :return: completed graph
        """
        batch_size, num_nodes, num_t = sample.shape[0], sample.shape[2], sample.shape[1]
        hidden_state, _ = self.initHidden(batch_size * num_nodes, self.hidden_size, device=sample.device)

        # Section: Long-term graph GRU encoding
        for current_t in range(0, num_t-4, 4):
            current_graph = sample[:, current_t]
            hidden_state = self.gru_step(current_graph, hidden_state, predefined_A)

        # Section: Short-term learning (joint spatiotemporal attention)
        atten_scores = []
        b_src_filter = sample[:, -4:]
        tar_filer = sample[:, -1]
        for i in range(self.TGN_layers):
            # beforehand temporal graph attention (include target graph)
            b_attn_scores = self.TGNs[i]['b_tga'](tar_filer, b_src_filter)
            atten_scores.append(b_attn_scores)
            b_src_filter = b_src_filter.reshape([-1, num_nodes, b_src_filter.shape[-1]])
            # merge attn scores into pre_defined adjacent matrix
            b_A_tgn = [predefined_A.unsqueeze(0) + b_attn_scores.reshape([-1, num_nodes, num_nodes])]
            b_src_filter = (
                    self.TGNs[i]['b_tgn1'](b_src_filter, b_A_tgn[0])
                    ).reshape([batch_size, b_src_filter.shape[0]//batch_size, num_nodes, -1])
            b_src_filter = torch.relu(b_src_filter)
            tar_filer = torch.sum(b_src_filter, dim=1)
        gat_result = self.fc_final_short(tar_filer).reshape([batch_size, num_nodes, -1])

        # Section: long results
        hidden_state = self.gru_step(gat_result, hidden_state, predefined_A)
        hidden_state = hidden_state.reshape([batch_size, num_nodes, -1])
        gru_result = self.fc_final_long(hidden_state)

        final_result = self.fc_final_mix(torch.cat([tar_filer, hidden_state], dim=2))

        return gat_result, gru_result, final_result, atten_scores[0]

    def gru_step(self, current_graph, hidden_state, predefined_A):
        """
        Kriging one time step (reference graph)
        :param: current_graph: current input for graph GRU [batch, num_nodes, num_features]
        :param: hidden_state:  [batch, num_nodes, hidden_size]
        :param: predefined_A: predefined adjacent matrix, static per iteration, no need batch [num_nodes, num_nodes]
        :return: kriging results of current reference graph
        """
        batch_size, num_nodes = current_graph.shape[0], current_graph.shape[1]
        hidden_state = hidden_state.view(-1, num_nodes, self.hidden_size)

        # Section: Generate graph for graph learning
        graph_source = self.GCN_agg1(hidden_state, predefined_A)
        graph_target = self.GCN_agg2(hidden_state, predefined_A)

        nodevec_source = torch.tanh(self.alpha * torch.mul(
            self.source_nodeemb(current_graph), graph_source))
        nodevec_target = torch.tanh(self.alpha * torch.mul(
            self.target_nodeemb(current_graph), graph_target))

        a = torch.matmul(nodevec_source, nodevec_target.transpose(2, 1)) \
            - torch.matmul(nodevec_target, nodevec_source.transpose(2, 1))

        adp_adj = torch.relu(torch.tanh(self.alpha * a))

        adp = self.adj_processing(adp_adj, num_nodes, predefined_A, current_graph.device)

        # Section: Long_term Learning
        combined = torch.cat((current_graph, hidden_state), -1)
        z = torch.sigmoid(self.gz1_de(combined, adp))
        r = torch.sigmoid(self.gr1_de(combined, adp))
        temp = torch.cat((current_graph, torch.mul(r, hidden_state)), dim=-1)
        cell_state = torch.tanh(self.gc1_de(temp, adp))
        hidden_state = torch.mul(z, hidden_state) + torch.mul(1 - z, cell_state)
        hidden_state = hidden_state.reshape([-1, self.hidden_size])

        return hidden_state

    def adj_processing(self, adp_adj, num_nodes, predefined_A, device):
        adp_adj = adp_adj + torch.eye(num_nodes).to(device)
        adp_adj = adp_adj / torch.unsqueeze(adp_adj.sum(-1), -1)
        return [adp_adj, predefined_A]

    def initHidden(self, batch_size, hidden_size, device):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(
                torch.zeros(batch_size, hidden_size).to(device))
            Cell_State = Variable(
                torch.zeros(batch_size, hidden_size).to(device))
            # nn.init.orthogonal_(Hidden_State)
            # nn.init.orthogonal_(Cell_State)
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, hidden_size))
            return Hidden_State, Cell_State

    def get_source_index(self, num_t, current_t):

        b_index = [i for i in range(max(0, current_t-2), current_t+1)]  # must has at least oen index
        a_index = [i for i in range(current_t+1, min(current_t+3, num_t))]  # might be empty
        return b_index, a_index


class GatedGCN(nn.Module):

    def __init__(self, dims, gdep, dropout, alpha, beta, gamma, type=None):
        super().__init__()
        self.gcn = GCN(dims, gdep, dropout, alpha, beta, gamma, type=type)
        self.gate_gcn = GCN(dims, gdep, dropout, alpha, beta, gamma, type=type)
        self.gcnT = GCN(dims, gdep, dropout, alpha, beta, gamma, type=type)
        self.gate_gcnT = GCN(dims, gdep, dropout, alpha, beta, gamma, type=type)


    def forward(self, input, adj, adjT):
        return torch.sigmoid(self.gate_gcn(input, adj) + self.gate_gcnT(input, adjT)) \
               * torch.tanh(self.gcn(input, adj) + self.gcnT(input, adjT))


class GCN(nn.Module):
    def __init__(self, dims, gdep, dropout, alpha, beta, gamma, type=None):
        super(GCN, self).__init__()
        if type == 'RNN':
            self.gconv = GconvAdp()
            self.gconv_preA = GconvPre()
            self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])

        elif type == 'hyper':
            self.gconv_preA = GconvPre()
            self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])

        elif type == 'TGN':
            self.gconv = GconvAdp()
            self.mlp = nn.Linear((gdep + 1) * dims[0], dims[1])
        else:
            raise NotImplementedError('GCN type is not implemented!')

        if dropout:
            self.dropout_ = nn.Dropout(p=dropout)

        self.dropout = dropout
        self.gdep = gdep
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.type_GNN = type

    def forward(self, x, adj):

        h = x
        out = [h]
        if self.type_GNN == 'RNN':
            for _ in range(self.gdep):
                h = self.alpha * x + self.beta * self.gconv(
                    h, adj[0]) + self.gamma * self.gconv_preA(h, adj[1])
                out.append(h)
        elif self.type_GNN == 'hyper':
            for _ in range(self.gdep):
                h = self.alpha * x + self.gamma * self.gconv_preA(h, adj)
                out.append(h)
        elif self.type_GNN == 'TGN':
            for _ in range(self.gdep):
                h = self.alpha * x + self.gamma * self.gconv(h, adj)
                out.append(h)

        ho = torch.cat(out, dim=-1)

        ho = self.mlp(ho)
        if self.dropout:
            ho = self.dropout_(ho)

        return ho


class GconvAdp(nn.Module):
    def __init__(self):
        super(GconvAdp, self).__init__()

    def forward(self, x, A):
        if x.shape[0] != A.shape[0]:
            A = A.unsqueeze(1).repeat(1, x.shape[0]//A.shape[0], 1, 1).reshape([-1, x.shape[1], x.shape[1]])
        x = torch.einsum('nvc,nvw->nwc', (x, A))
        return x.contiguous()


class GconvPre(nn.Module):
    def __init__(self):
        super(GconvPre, self).__init__()

    def forward(self, x, A):

        x = torch.einsum('nvc,vw->nwc', (x, A))
        return x.contiguous()


class MultiAtten(nn.Module):
    def __init__(self):
        super(MultiAtten, self).__init__()


class TAttn(nn.Module):

    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in * 4
        self.w_query = nn.Parameter(torch.FloatTensor(dim_in, dim_out))
        self.w_key = nn.Parameter(torch.FloatTensor(dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        self.trans = nn.Parameter(torch.FloatTensor(dim_out, 1))

    def forward(self, query, keys):
        """

        :param query: current kriging graph [batch, num_node, num_features_2]
        :param keys: graphs in the temporal direction [batch, num_time, num_node, num_features_1]
        :return: temporal attention scores (a.k.a temporal attention adjacent matrix) [batch, num_time, num_node, num_node]
        """

        query = torch.matmul(query, self.w_query).unsqueeze(1).unsqueeze(3)
        keys = torch.matmul(keys, self.w_key).unsqueeze(2)
        attn_scores = torch.matmul(torch.tanh(query + keys + self.bias), self.trans).squeeze(-1)
        # multi_dimensional Softmax
        attn_scores = torch.exp(attn_scores) / torch.sum(torch.exp(attn_scores), dim=-1, keepdim=True)

        return attn_scores
