from enum import Enum

import torch.optim as optim


class Optimizer(Enum):
    """
    Class representing the optimizer options.
    """
    adam = 'adam'
    sgd = "sdg"
    rmsprop = "rmsprop"


class OptimizerBase:
    def __init__(self, params, lr):

        self.params = params
        self.lr = lr

    def optimizer(self, optimizer: Optimizer):
        if optimizer == Optimizer.adam.value:
            return optim.Adam(self.params, lr=self.lr)

        elif optimizer == Optimizer.sgd.value:
            return optim.SGD(params=self.params, lr=self.lr)

        elif optimizer == Optimizer.rmsprop.value:
            return optim.RMSprop(self.params, lr=self.lr)
