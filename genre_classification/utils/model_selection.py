from genre_classification.data_model.tl_models import TLModel
from genre_classification.model.tl_alexnet_model import train_alex_net_model
from genre_classification.model.tl_densenet121_model import train_densenet121_model
from genre_classification.model.tl_resnet18_model import train_resnet18_model
from genre_classification.model.tl_resnet34_model import train_resnet34_model
from genre_classification.model.tl_vgg16_model import train_vgg_model


def model_selection(tl_model, criterion, optimizer):
    if tl_model is TLModel.alexnet.value:
        return train_alex_net_model(criterion=criterion, optimizer=optimizer)
    elif tl_model is TLModel.resnet18.value:
        return train_resnet18_model(criterion=criterion, optimizer=optimizer)
    elif tl_model is TLModel.resnet34.value:
        return train_resnet34_model(criterion=criterion, optimizer=optimizer)
    elif tl_model is TLModel.vgg.value:
        return train_vgg_model(criterion=criterion, optimizer=optimizer)
    elif tl_model is TLModel.densenet121.value:
        return train_densenet121_model(criterion=criterion, optimizer=optimizer)
    else:
        raise Exception
