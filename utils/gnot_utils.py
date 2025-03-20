import matplotlib.pyplot as plt
import numpy as np
import operator
from functools import reduce
import dgl
from dgl.nn.pytorch import SumPooling, AvgPooling
import torch
from torch.nn.modules.loss import _WeightedLoss
from torch.nn.utils.rnn import pad_sequence
from collections.abc import Iterable


################################################################
#  GNOT
################################################################

### x: list of tensors
class MultipleTensors():
    """Source code: https://github.com/HaoZhongkai/GNOT/blob/master/utils.py"""
    def __init__(self, x):
        self.x = x

    def to(self, device):
        self.x = [x_.to(device) for x_ in self.x]
        return self

    def __len__(self):
        return len(self.x)


    def __getitem__(self, item):
        return self.x[item]
    
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
  
class WeightedLpRelLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, component=0,regularizer=False, normalizer=None):
        super(WeightedLpRelLoss, self).__init__()

        self.d = d
        self.p = p
        self.component = component if component == 'all' or 'all-reduce' else int(component)
        self.regularizer = regularizer
        self.normalizer = normalizer
        self.sum_pool = SumPooling()

    ### all reduce is used in temporal cases, use only one metric for all components
    def _lp_losses(self, g, pred, target):
        if (self.component == 'all') or (self.component == 'all-reduce'):
            err_pool = (self.sum_pool(g, (pred - target).abs() ** self.p))
            target_pool = (self.sum_pool(g, target.abs() ** self.p))
            losses = (err_pool / target_pool)**(1/ self.p)
            if self.component == 'all':
                metrics = losses.mean(dim=0).clone().detach().cpu().numpy()
            else:
                metrics = losses.mean().clone().detach().cpu().numpy()

        else:
            assert self.component <= target.shape[1]
            err_pool = (self.sum_pool(g, (pred - target[:,self.component]).abs() ** self.p))
            target_pool = (self.sum_pool(g, target[:,self.component].abs() ** self.p))
            losses = (err_pool / target_pool)**(1/ self.p)
            metrics = losses.mean().clone().detach().cpu().numpy()

        loss = losses.mean()

        return loss, metrics

    def forward(self, g,  pred, target):

        #### only for computing metrics


        loss, metrics = self._lp_losses(g, pred, target)

        if self.normalizer is not None:
            ori_pred, ori_target = self.normalizer.transform(pred,component=self.component,inverse=True), self.normalizer.transform(target, inverse=True)
            _, metrics = self._lp_losses(g, ori_pred, ori_target)

        if self.regularizer:
            raise NotImplementedError
        else:
            reg = torch.zeros_like(loss)


        return loss, reg, metrics


class WeightedLpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, component=0, regularizer=False, normalizer=None):
        super(WeightedLpLoss, self).__init__()

        self.d = d
        self.p = p
        self.component = component if component == 'all' else int(component)
        self.regularizer = regularizer
        self.normalizer = normalizer
        self.avg_pool = AvgPooling()

    def _lp_losses(self, g, pred, target):
        if self.component == 'all':
            losses = self.avg_pool(g, ((pred - target).abs() ** self.p)) ** (1 / self.p)
            metrics = losses.mean(dim=0).clone().detach().cpu().numpy()

        else:
            assert self.component <= target.shape[1]
            losses = self.avg_pool(g, (pred - target[:, self.component]).abs() ** self.p) ** (1 / self.p)
            metrics = losses.mean().clone().detach().cpu().numpy()

        loss = losses.mean()

        return loss, metrics

    def forward(self, g, pred, target):

        #### only for computing metrics

        loss, metrics = self._lp_losses(g, pred, target)

        if self.normalizer is not None:
            ori_pred, ori_target = self.normalizer.transform(pred,component=self.component, inverse=True), self.normalizer.transform(target, inverse=True)
            _, metrics = self._lp_losses(g, ori_pred, ori_target)

        if self.regularizer:
            raise NotImplementedError
        else:
            reg = torch.zeros_like(loss)

        return loss, reg, metrics  

def get_loss_func(name, args, **kwargs):
    if name == 'rel2':
        return WeightedLpRelLoss(p=2,component=args.component, normalizer=kwargs['normalizer'])
    elif name == "rel1":
        return WeightedLpRelLoss(p=1,component=args.component, normalizer=kwargs['normalizer'])
    elif name == 'l2':
        return WeightedLpLoss(p=2, component=args.component, normalizer=kwargs["normalizer" ])
    elif name == "l1":
        return WeightedLpLoss(p=1, component=args.component, normalizer=kwargs["normalizer" ])
    else:
        raise NotImplementedError

def get_num_params(model):
    '''
    a single entry in cfloat and cdouble count as two parameters
    see https://github.com/pytorch/pytorch/issues/57518
    '''
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # num_params = 0
    # for p in model_parameters:
    #     # num_params += np.prod(p.size()+(2,) if p.is_complex() else p.size())
    #     num_params += p.numel() * (1 + p.is_complex())
    # return num_params

    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))  #### there is complex weight
    return c    

'''
    Simple normalization layer
'''
class UnitTransformer():
    def __init__(self, X):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True) + 1e-8


    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, X, inverse=True,component='all'):
        if component == 'all' or 'all-reduce':
            if inverse:
                orig_shape = X.shape
                return (X*(self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X-self.mean)/self.std
        else:
            if inverse:
                orig_shape = X.shape
                return (X*(self.std[:,component] - 1e-8)+ self.mean[:,component]).view(orig_shape)
            else:
                return (X - self.mean[:,component])/self.std[:,component]


'''
    Simple pointwise normalization layer, all data must contain the same length, used only for FNO datasets
    X: B, N, C
'''
class PointWiseUnitTransformer():
    def __init__(self, X):
        self.mean = X.mean(dim=0, keepdim=False)
        self.std = X.std(dim=0, keepdim=False) + 1e-8


    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, X, inverse=True,component='all'):
        if component == 'all' or 'all-reduce':
            if inverse:
                orig_shape = X.shape
                X = X.view(-1, self.mean.shape[0],self.mean.shape[1])   ### align shape for flat tensor
                return (X*(self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X-self.mean)/self.std
        else:
            if inverse:
                orig_shape = X.shape
                X = X.view(-1, self.mean.shape[0],self.mean.shape[1])
                return (X*(self.std[:,component] - 1e-8)+ self.mean[:,component]).view(orig_shape)
            else:
                return (X - self.mean[:,component])/self.std[:,component]