from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class SequentialDataSet(Dataset, ABC):
    @abstractmethod
    def input_shape(self):
        '''Returns a tuple input shape of the dataset (L, S, D), where:
        L = spatial dimension length
        S = number state variables
        D = number of spatial dimensions 
        :return: tuple
        '''
        raise NotImplementedError
