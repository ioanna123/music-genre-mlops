from torchvision import models

from genre_classification.data_model.criterion import Criterion
from genre_classification.model.base_model import TLModelBase
from genre_classification.trainer.optimizer import Optimizer


class Densenet121Model(TLModelBase):

    def __init__(self, model=models.densenet121(pretrained=True),
                 criterion=Criterion.cross_entropy.value,
                 optimizer=Optimizer.adam.value):
        super().__init__(model=model, criterion=criterion, optimizer=optimizer)
