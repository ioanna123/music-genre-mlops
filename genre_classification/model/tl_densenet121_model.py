from torchvision import models

from genre_classification.data_model.criterion import Criterion
from genre_classification.model.base_model import TLModelBase
from genre_classification.trainer.optimizer import Optimizer


class Densenet121Model(TLModelBase):

    def __init__(self, model, criterion, optimizer, features):
        super().__init__(model=model, criterion=criterion, optimizer=optimizer,
                         in_features=features)


def train_densenet121_model(criterion: Criterion,
                            optimizer: Optimizer,
                            model=models.densenet121(pretrained=True)) -> Densenet121Model:
    return Densenet121Model(model, criterion, optimizer, features=model.classifier.in_features)
