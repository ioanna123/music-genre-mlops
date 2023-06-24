from enum import Enum

from torch import nn


class Criterion(Enum):
    """
    Class representing the criterion options.
    """
    cross_entropy = nn.CrossEntropyLoss()
    kldiv_loss = nn.KLDivLoss()
    smooth_loss = nn.SmoothL1Loss()
