from enum import Enum

from torch import nn


class Criterion(Enum):
    """
    Class representing the criterion options.
    """
    cross_entropy = nn.CrossEntropyLoss()
    kldiv_loss = nn.KLDivLoss()


def return_criterion(criterion_val) -> nn:
    if criterion_val == Criterion.cross_entropy.name:
        return Criterion.cross_entropy.value
    elif criterion_val == Criterion.kldiv_loss.name:
        return Criterion.kldiv_loss.value
    else:
        raise Exception(f'Unsupported criterion: {criterion_val}')
