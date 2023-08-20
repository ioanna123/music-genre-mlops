from torchvision import models

from genre_classification.data_model.criterion import Criterion
from genre_classification.model.base_model import TLModelBase
from genre_classification.trainer.optimizer import Optimizer


class Resnet34Model(TLModelBase):

    def __init__(self, model, criterion, optimizer, features):
        super().__init__(model_name='resnet34', model=model, criterion=criterion, optimizer=optimizer,
                         in_features=features)


def train_resnet34_model(criterion: Criterion,
                         optimizer: Optimizer,
                         model=models.resnet34(pretrained=True)) -> Resnet34Model:
    return Resnet34Model(model, criterion, optimizer, features=model.fc.in_features)
