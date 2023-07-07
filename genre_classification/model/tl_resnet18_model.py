from torchvision import models

from genre_classification.data_model.criterion import Criterion
from genre_classification.model.base_model import TLModelBase
from genre_classification.trainer.optimizer import Optimizer


class Resnet18Model(TLModelBase):

    def __init__(self, model, criterion, optimizer):
        super().__init__(model=model, criterion=criterion, optimizer=optimizer)


def train_resnet18_model(model=models.resnet18(pretrained=True),
                         criterion=Criterion.cross_entropy.value,
                         optimizer=Optimizer.adam.value) -> Resnet18Model:
    return Resnet18Model(model, criterion, optimizer)
