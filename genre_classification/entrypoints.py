from os import listdir
from os.path import isfile, join

from genre_classification.data_model.criterion import Criterion
from genre_classification.data_model.tl_models import TLModel
from genre_classification.feature_extraction.factories import get_feature_extraction
from genre_classification.feature_extraction.feature_extraction import FeatureExtraction
from genre_classification.preprocessor.audio_preprocess import AudioPreprocess
from genre_classification.preprocessor.factories import get_dataset, get_audio_preprocessor
from genre_classification.trainer.optimizer import Optimizer
from genre_classification.utils.metadata import extract_audio_metadata
from genre_classification.utils.model_selection import model_selection
from genre_classification.utils.save_mel_spec_img import save_mel_spec_per_genre
from settings import window_duration


def train_tl_model_images(tl_model: TLModel, criterion: Criterion, optimizer: Optimizer,
                          checkpoints_path: str, images_path: str, save: bool = True, num_epoch: int = 10):
    """
    Starts a new end-to-end training given a model, a criterion, optimizer, image_data and save results to checkpoint path
    @param tl_model: The dl model for the training
    @param criterion: The criterion for the training
    @param optimizer: The optimizer for the training
    @param checkpoints_path: path to save the models checkpoint
    @param images_path: path to load the featured images for training
    @param save: if true save the checkpoints to desired path
    @param num_epoch: the num of epochs for training
    """
    model = model_selection(tl_model, criterion, optimizer)

    image_data_loader = get_dataset().transform(images_path)

    trained_model, train_losses, val_losses = model.train(
        train_dataloader=image_data_loader.train_dataloader,
        test_dataloader=image_data_loader.val_dataloader,
        save=save,
        num_epoch=num_epoch,
        checkpoint_path=checkpoints_path
    )

    classes = image_data_loader.train_dataloader.dataset.dataset.classes
    precision, recall, fscore, support = model.evaluate_model(
        test_subset=image_data_loader.test_subset, model=trained_model, classes=classes)

    return precision, recall, fscore, support


def create_image_features_from_audio(path_with_audios_dir: str, path_to_image: str,
                                     preprocessor: AudioPreprocess = None, feature_extractor: FeatureExtraction = None):
    preprocessor = preprocessor if preprocessor is not None else get_audio_preprocessor()
    feature_extractor = feature_extractor if feature_extractor is not None else get_feature_extraction()
    audio_dirs_genre = [join(path_with_audios_dir, f) for f in listdir(path_with_audios_dir) if
                        isfile(join(path_with_audios_dir, f)) or listdir(join(path_with_audios_dir, f))]
    for audio_genre in audio_dirs_genre:
        audio_files = [join(audio_genre, f) for f in listdir(audio_genre) if
                       isfile(join(audio_genre, f))]

        for audio in audio_files:
            meta = extract_audio_metadata(audio)
            for start in range(0, int(meta.duration), window_duration):
                for streamed in preprocessor.stream(audio, start=start, window_duration=window_duration):
                    features = feature_extractor.transform(streamed)
                    save_mel_spec_per_genre(
                        image_dir=path_to_image,
                        image_name=audio.split('.')[1],
                        mel_spec=features,
                        genre=audio.split('.')[0].split('/')[-1]
                    )


def train_tl_model_audio(tl_model: TLModel, criterion: Criterion, optimizer: Optimizer,
                         checkpoints_path: str, save_images_path: str, audio_paths: str, save: bool = True,
                         num_epoch: int = 10):
    """
    Starts a new end-to-end training given a model, a criterion, optimizer, image_data and save results to checkpoint path
    @param tl_model: The dl model for the training
    @param criterion: The criterion for the training
    @param optimizer: The optimizer for the training
    @param checkpoints_path: path to save the models checkpoint
    @param save_images_path: path to save the featured images for training
    @param save: if true save the checkpoints to desired path
    @param num_epoch: the num of epochs for training
    @param audio_paths: The audio paths eg Data/genre_originals
    """

    # model selection
    model = model_selection(tl_model, criterion, optimizer)

    # create image features
    create_image_features_from_audio(audio_paths, path_to_image=save_images_path)

    image_data_loader = get_dataset().transform(save_images_path)

    trained_model, train_losses, val_losses = model.train(
        train_dataloader=image_data_loader.train_dataloader,
        test_dataloader=image_data_loader.val_dataloader,
        save=save,
        num_epoch=num_epoch,
        checkpoint_path=checkpoints_path
    )
    classes = image_data_loader.train_dataloader.dataset.dataset.classes
    return model.evaluate_model(
        test_subset=image_data_loader.test_subset, model=trained_model, classes=classes)
