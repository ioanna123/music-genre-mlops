from genre_classification.feature_extraction.factories import get_feature_extraction
from genre_classification.preprocessor.factories import get_audio_preprocessor, get_dataset
from genre_classification.utils.metadata import extract_audio_metadata
from genre_classification.utils.save_mel_spec_img import save_mel_spec_per_genre
from settings import window_duration

feature_extractor = get_feature_extraction()
img_data = get_dataset()
preprocessor = get_audio_preprocessor()
if __name__ == "__main__":
    # e = img_data.transform('Data/images_original')
    # r_m = Resnet18Model()
    # r_m = Resnet34Model()
    # r_m = VGGModel()
    # r_m = AlexNetModel()
    # resnet, train_losses, val_losses = r_m.train(
    #     train_dataloader=e.train_dataloader,
    #     test_dataloader=e.val_dataloader,
    # )
    #
    # plt.plot(train_losses, label='Training loss')
    # plt.plot(val_losses, label='Validation loss')
    # plt.legend(frameon=False)
    # plt.show()

    test_audio = 'Data/genres_original/metal/metal.00043.wav'

    meta = extract_audio_metadata(test_audio)

    segments = []
    for start in range(0, int(meta.duration), window_duration):
        for streamed in preprocessor.stream(test_audio, start=start, window_duration=window_duration):
            features = feature_extractor.transform(streamed)
            print(features.value)
            save_mel_spec_per_genre(
                image_dir='image_data',
                image_name='00043',
                genre='metal_genre',
                mel_spec=features
            )
