import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

import settings as ml_setting
from genre_classification.data_model.dataset import DatasetLoader


class ImageDataset:
    def __init__(self):
        self.split_val_test = ml_setting.SPLIT_VAL_TEST
        self.split_train = ml_setting.SPLIT_TRAIN
        self.mean = ml_setting.MEAN
        self.std = ml_setting.STD
        self.batch_size = ml_setting.BATCH_SIZE

    def get_transforms(self):
        return transforms.Compose([
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    @staticmethod
    def load_data(path_imgs, transform):
        return datasets.ImageFolder(path_imgs, transform=transform)

    @staticmethod
    def permute_data(num_train_samples: int):
        return torch.randperm(num_train_samples)

    @staticmethod
    def data_subset(dataset, indices, start: int = None, end: int = None):
        return Subset(dataset, indices[start:end])

    def transform(self, path_images: str) -> DatasetLoader:
        transform = self.get_transforms()
        dataset = self.load_data(path_images, transform)
        train_split = int(len(dataset) * self.split_val_test)
        val_split = int(train_split * self.split_train)

        indices = self.permute_data(len(dataset))

        train_subset = self.data_subset(dataset, indices, start=train_split)
        val_subset = self.data_subset(dataset, indices, start=val_split, end=train_split)
        test_subset = self.data_subset(dataset, indices, end=val_split)

        train_dataloader = DataLoader(
            dataset=train_subset,
            batch_size=self.batch_size,
            shuffle=True
        )

        val_dataloader = torch.utils.data.DataLoader(
            dataset=val_subset,
            batch_size=self.batch_size,
            shuffle=False)

        classes = train_dataloader.dataset.dataset.classes

        return DatasetLoader(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_subset=test_subset,
            classes=classes
        )
