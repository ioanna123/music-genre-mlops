import genre_classification.entrypoints as ml_entrypoints
from genre_classification.data_model.criterion import Criterion
from genre_classification.data_model.tl_models import TLModel
from genre_classification.feature_extraction.factories import get_feature_extraction
from genre_classification.preprocessor.factories import get_audio_preprocessor, get_dataset
from genre_classification.trainer.optimizer import Optimizer

feature_extractor = get_feature_extraction()
img_data = get_dataset()
preprocessor = get_audio_preprocessor()

if __name__ == "__main__":
    # check create image features using audio files

    # ml_entrypoints.create_image_features_from_audio(path_with_audios_dir='Data/genres_original',
    #                                                 path_to_image='Test')

    # check train using original audios
    ml_entrypoints.train_tl_model_audio(tl_model=TLModel.resnet18.value, criterion=Criterion.cross_entropy.name,
                                        optimizer=Optimizer.adam.value,
                                        checkpoints_path="checkpoints", num_epoch=1,
                                        audio_paths='NewData', save_images_path='Test')
