import os
import torch
import numpy as np
import networkx as nx
import tqdm
import time
import pickle
import gc
import dgl

from sklearn.preprocessing import QuantileTransformer

from dgl.data import DGLDataset
from dgl.nn.pytorch import SumPooling, AvgPooling

from scipy import interpolate
from scipy.io import loadmat
from scipy.sparse import csr_matrix, diags
from torch.utils.data import Dataset
from torch.nn.modules.loss import _WeightedLoss
from torch.nn.utils.rnn import pad_sequence
from utils.gnot_utils import MultipleTensors, PointWiseUnitTransformer, UnitTransformer

'''
    A simple interface for processing FNO dataset,
    1. Data might be 1d, 2d, 3d
    2. X: concat of [pos, a], , we directly reshape them into a B*N*C array
    2. We could use pointwise normalizer since dimension of data is the same
    3. Building graphs for FNO dataset is fast since there is no edge info, we do not use cache
    4. for FNO dataset, we augment g_u = g and set u_p = 0
    
'''
class FNODataset(DGLDataset):
    """Source code: https://github.com/HaoZhongkai/GNOT/blob/master/data_utils.py"""
    def __init__(self, X, Y, name=' ',train=True,test=False, normalize_y=False, y_normalizer=None, normalize_x = False):
        self.normalize_y = normalize_y
        self.normalize_x = normalize_x
        self.y_normalizer = y_normalizer

        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(Y)


        ####  debug timing


        super(FNODataset, self).__init__(name)   #### invoke super method after read data



    def process(self):

        self.data_len = len(self.x_data)
        self.n_dim = self.x_data.shape[1]
        self.graphs = []
        self.graphs_u = []
        self.u_p = []
        for i in range(len(self)):
            x_t, y_t = self.x_data[i].float(), self.y_data[i].float()
            g = dgl.DGLGraph()
            g.add_nodes(self.n_dim)
            g.ndata['x'] = x_t
            g.ndata['y'] = y_t
            up = torch.zeros([1])
            u = torch.zeros([1])
            u_flag = torch.zeros(g.number_of_nodes(),1)
            g.ndata['u_flag'] = u_flag
            self.graphs.append(g)
            self.u_p.append(up) # global input parameters
            g_u = dgl.DGLGraph()
            g_u.add_nodes(self.n_dim)
            g_u.ndata['x'] = x_t
            g_u.ndata['u'] = torch.zeros(g_u.number_of_nodes(), 1)

            self.graphs_u.append(g_u)

            # print('processing {}'.format(i))

        self.u_p = torch.stack(self.u_p)


        #### normalize_y
        if self.normalize_y:
            self.__normalize_y()
        if self.normalize_x:
            self.__normalize_x()

        return

    def __normalize_y(self):
        if self.y_normalizer is None:

            self.y_normalizer = PointWiseUnitTransformer(self.y_data)
            # print('point wise normalizer shape',self.y_normalizer.mean.shape, self.y_normalizer.std.shape)

            # y_feats_all = torch.cat([g.ndata['y'] for g in self.graphs],dim=0)
            # self.y_normalizer = UnitTransformer(y_feats_all)


        for g in self.graphs:
            g.ndata['y'] = self.y_normalizer.transform(g.ndata["y"], inverse=False)  # a torch quantile transformer

        print('Target features are normalized using pointwise unit normalizer')
        # print('Target features are normalized using unit transformer')


    def __normalize_x(self):
        x_feats_all = torch.cat([g.ndata["x"] for g in self.graphs],dim=0)

        self.x_normalizer = UnitTransformer(x_feats_all)

        # for g in self.graphs:
        #     g.ndata['x'] = self.x_normalizer.transform(g.ndata['x'], inverse=False)

        # if self.graphs_u[0].number_of_nodes() > 0:
        #     for g in self.graphs_u:
        #         g.ndata['x'] = self.x_normalizer.transform(g.ndata['x'], inverse=False)

        self.up_normalizer = UnitTransformer(self.u_p)
        self.u_p = self.up_normalizer.transform(self.u_p, inverse=False)


        print('Input features are normalized using unit transformer')


    def __len__(self):
        return self.data_len


    def __getitem__(self, idx):
        return self.graphs[idx], self.u_p[idx], self.graphs_u[idx]



def collate_op(items):
    transposed = zip(*items)
    batched = []
    for sample in transposed:
        if isinstance(sample[0], dgl.DGLGraph):
            batched.append(dgl.batch(list(sample)))
        elif isinstance(sample[0], torch.Tensor):
            batched.append(torch.stack(sample))
        elif isinstance(sample[0], MultipleTensors):
            sample_ = MultipleTensors([pad_sequence([sample[i][j] for i in range(len(sample))]).permute(1,0,2) for j in range(len(sample[0]))])
            batched.append(sample_)
        else:
            raise NotImplementedError
    return batched