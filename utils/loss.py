import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy
import os
from datetime import datetime
import operator
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.modules.loss import _WeightedLoss
from torch.nn.utils.rnn import pad_sequence
from collections.abc import Iterable

class RelativeL2Loss(nn.Module):
    def __init__(self, size_average=True, reduction=True):
        """
        Relative L2 Loss: ||x-y||₂/||y||₂
        
        Args:
            size_average (bool): If True, returns the mean of relative errors.
                                If False, returns the sum of relative errors.
            reduction (bool): If False, returns individual errors without reduction.
        """
        super(RelativeL2Loss, self).__init__()
        self.size_average = size_average
        self.reduction = reduction
    
    def forward(self, x, y):
        """
        Compute the relative L2 error between predictions and targets.
        
        Args:
            x (torch.Tensor): Prediction tensor
            y (torch.Tensor): Target tensor
            
        Returns:
            torch.Tensor: Relative L2 error
        """
        num_examples = x.size()[0]
        
        # Compute L2 norm of the difference (p=2)
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 2, 1)
        
        # Compute L2 norm of the target
        y_norms = torch.norm(y.reshape(num_examples, -1), 2, 1)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        rel_errors = diff_norms / (y_norms + eps)
        
        if self.reduction:
            if self.size_average:
                return torch.mean(rel_errors)
            else:
                return torch.sum(rel_errors)
        return rel_errors       