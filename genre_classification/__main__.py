import click

import genre_classification.entrypoints as ml_entrypoints
from genre_classification.data_model.criterion import Criterion
from genre_classification.data_model.tl_models import TLModel
from genre_classification.feature_extraction.feature_extraction import FeatureExtraction
from genre_classification.preprocessor.audio_preprocess import AudioPreprocess
from genre_classification.trainer.optimizer import Optimizer


@click.group()
def cli():
    pass


@cli.command()
@click.option('--model', type=click.Choice([str(TLModel.vgg.value), str(TLModel.alexnet.value),
                                            str(TLModel.densenet121.value), str(TLModel.resnet34.value),
                                            str(TLModel.resnet18.value)]))
@click.option('--criterion', type=click.Choice([str(Criterion.cross_entropy.value), str(Criterion.kldiv_loss.value),
                                                str(Criterion.smooth_loss.value)]))
@click.option('--optimizer', type=click.Choice([str(Optimizer.adam.value), str(Optimizer.sgd.value),
                                                str(Optimizer.rmsprop.value)]))
@click.option('--checkpoints_path', type=click.STRING, required=True, help='Checkpoint path to save models')
@click.option('--images_path', type=click.STRING, required=True, help='Path to load the featured images for training')
@click.option('--save', type=click.BOOL, required=False, default=True,
              help='if true save the checkpoints to desired path')
@click.option('--num_epoch', type=click.INT, default=10, help='The num of epochs for training')
def train_using_image_features(model: TLModel, criterion: Criterion, optimizer: Optimizer,
                               checkpoints_path: str, images_path: str, save: bool, num_epoch: int):
    ml_entrypoints.train_tl_model_images(tl_model=model, criterion=criterion, optimizer=optimizer,
                                         checkpoints_path=checkpoints_path, images_path=images_path, save=save,
                                         num_epoch=num_epoch)


@cli.command()
@click.option('--model', type=click.Choice([str(TLModel.vgg.value), str(TLModel.alexnet.value),
                                            str(TLModel.densenet121.value), str(TLModel.resnet34.value),
                                            str(TLModel.resnet18.value)]))
@click.option('--criterion', type=click.Choice([str(Criterion.cross_entropy.value), str(Criterion.kldiv_loss.value),
                                                str(Criterion.smooth_loss.value)]))
@click.option('--optimizer', type=click.Choice([str(Optimizer.adam.value), str(Optimizer.sgd.value),
                                                str(Optimizer.rmsprop.value)]))
@click.option('--checkpoints_path', type=click.STRING, required=True, help='Checkpoint path to save models')
@click.option('--save_images_path', type=click.STRING, required=True,
              help='Path to save the featured images for training')
@click.option('--audio_paths', type=click.STRING, required=True, help='The audio paths eg Data/genre_originals')
@click.option('--num_epoch', type=click.INT, default=10, help='The num of epochs for training')
@click.option('--save', type=click.BOOL, required=False, default=True)
def train_using_original_audios(model: TLModel, criterion: Criterion, optimizer: Optimizer,
                                checkpoints_path: str, save_images_path: str, audio_paths: str, save: bool,
                                num_epoch: int):
    ml_entrypoints.train_tl_model_audio(tl_model=model, criterion=criterion, optimizer=optimizer,
                                        checkpoints_path=checkpoints_path, save_images_path=save_images_path, save=save,
                                        num_epoch=num_epoch, audio_paths=audio_paths)


@cli.command()
@click.option('--path_with_audios_genre_dir', type=click.STRING, required=True)
@click.option('--path_to_image_genre', type=click.STRING, required=True)
def create_image_features_from_audio(path_with_audios_genre_dir: str, path_to_image_genre: str,
                                     preprocessor: AudioPreprocess = None, feature_extractor: FeatureExtraction = None):
    ml_entrypoints.create_image_features_from_audio(path_with_audios_genre_dir=path_with_audios_genre_dir,
                                                    path_to_image_genre=path_to_image_genre, preprocessor=preprocessor,
                                                    feature_extractor=feature_extractor)


if __name__ == '__main__':
    cli()
