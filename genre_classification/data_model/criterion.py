from enum import Enum

from torch import nn


class Criterion(Enum):
    """
    Class representing the criterion options.
    """
    cross_entropy = nn.CrossEntropyLoss()
    kldiv_loss = nn.KLDivLoss()
    smooth_loss = nn.SmoothL1Loss()


def return_criterion(criterion_val) -> nn:
    if criterion_val == Criterion.smooth_loss.name:
        return Criterion.smooth_loss.value
    elif criterion_val == Criterion.cross_entropy.name:
        return Criterion.cross_entropy.value
    elif criterion_val == Criterion.kldiv_loss.name:
        return Criterion.kldiv_loss.value
    else:
        raise Exception(f'Unsupported criterion: {criterion_val}')
