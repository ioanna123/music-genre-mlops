from torchvision import models

from genre_classification.data_model.criterion import Criterion
from genre_classification.model.base_model import TLModelBase
from genre_classification.trainer.optimizer import Optimizer


class Resnet34Model(TLModelBase):

    def __init__(self, model, criterion, optimizer):
        super().__init__(model=model, criterion=criterion, optimizer=optimizer)


def train_resnet34_model(model=models.resnet34(pretrained=True),
                         criterion=Criterion.cross_entropy.value,
                         optimizer=Optimizer.adam.value) -> Resnet34Model:
    return Resnet34Model(model, criterion, optimizer)
