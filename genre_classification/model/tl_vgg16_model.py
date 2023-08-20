from torchvision import models

from genre_classification.data_model.criterion import Criterion
from genre_classification.model.base_model import TLModelBase
from genre_classification.trainer.optimizer import Optimizer


class VGGModel(TLModelBase):

    def __init__(self, model, criterion, optimizer, features):
        super().__init__(model_name='vgg16', model=model, criterion=criterion, optimizer=optimizer,
                         in_features=features)


def train_vgg_model(criterion: Criterion,
                    optimizer: Optimizer,
                    model=models.vgg16(pretrained=True)) -> VGGModel:
    return VGGModel(model, criterion, optimizer, features=model.classifier[-1].in_features)
