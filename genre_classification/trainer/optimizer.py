from enum import Enum
from typing import Tuple, Union

import torch.optim as optim


class Optimizer(Enum):
    """
    Class representing the optimizer options.
    """
    adam = 'adam'
    sdg = "sdg"
    rmsprop = "rmsprop"


class OptimizerBase:
    def __init__(self, params, lr):

        self.params = params
        self.lr = lr

    def optimizer(self, optimizer: Optimizer):
        if optimizer == Optimizer.adam.value:
            return optim.Adam(self.params, lr=self.lr)

        elif optimizer == Optimizer.sdg.value:
            return optim.SGD(params=self.params, lr=self.lr)

        elif optimizer == Optimizer.rmsprop.value:
            return optim.RMSprop(self.params, lr=self.lr)


def suggest_optimizer(trial) -> Tuple[Union[str, float], str]:
    # Learning rate on a logarithmic scale
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    # Optimizer to use as categorical value
    optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "Adadelta", "sdg", "rmsprop"])

    return lr, optimizer_name
