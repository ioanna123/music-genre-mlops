from enum import Enum


class TLModel(Enum):
    """
    Class representing the criterion options.
    """
    alexnet = 'alexnet'
    densenet121 = 'densenet121'
    resnet18 = 'resnet18'
    resnet34 = 'resnet34'
    vgg = 'vgg'
