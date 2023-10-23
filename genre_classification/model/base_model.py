from abc import ABC
from time import time

import mlflow
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support as score
from torch import nn
from tqdm import tqdm

from genre_classification.data_model.criterion import Criterion
from genre_classification.data_model.evaluation import EvaluationMetrics
from genre_classification.trainer.optimizer import Optimizer, OptimizerBase
from settings import LR


class TLModelBase(ABC):

    def __init__(self,
                 model,
                 criterion: Criterion,
                 optimizer: Optimizer,
                 in_features: int,
                 n_classes: int = 10,
                 device='cuda' if torch.cuda.is_available() else 'cpu'
                 ):
        self.model = model
        self.n_classes = n_classes
        self.device = device
        self.criterion = criterion
        self.optimizer_option = optimizer
        self.in_features = in_features

        self.genre_dict = {"blues": 0, "classical": 1, "country": 2, "disco": 3, "hiphop": 4, "jazz": 5, "metal": 6,
                           "pop": 7, "reggae": 8, "rock": 9}

        # Fix the trainable parameters
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Replacing the Last Fully Connected Layer
        fc = nn.Linear(in_features=self.in_features, out_features=self.n_classes)
        self.model.fc = fc

    def train(self, test_dataloader, train_dataloader, num_epoch: int = 10):
        with mlflow.start_run():
            steps = 0
            train_losses, val_losses = [], []
            device = self.device
            mlflow.log_param('device', device)
            self.model.to(device)
            for param in self.model.parameters():
                param.requires_grad = True

            optimizer = OptimizerBase(self.model.parameters(), LR).optimizer(self.optimizer_option)
            for epoch in tqdm(range(num_epoch)):
                running_loss = 0
                correct_train = 0
                total_train = 0
                iter_time = time()

                self.model.train()

                for i, (images, labels) in enumerate(train_dataloader):
                    steps += 1
                    images = images.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    output = self.model(images)
                    loss = self.criterion(output, labels)

                    correct_train += (torch.max(output, dim=1)[1] == labels).sum()
                    total_train += labels.size(0)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    # Logging
                    if steps % 1 == 0:
                        # if True:
                        print(f'Epoch [{epoch + 1}]/[{num_epoch}]. Batch [{i + 1}]/[{len(train_dataloader)}].',
                              end=' ')
                        print(f'Train loss {running_loss / steps:.3f}.', end=' ')
                        print(f'Train acc {correct_train / total_train * 100:.3f}.', end=' ')
                        with torch.no_grad():
                            self.model.eval()
                            correct_val, total_val = 0, 0
                            val_loss = 0
                            for images, labels in test_dataloader:
                                images = images.to(device)
                                labels = labels.to(device)
                                output = self.model(images)
                                loss = self.criterion(output, labels)
                                val_loss += loss.item()

                                correct_val += (torch.max(output, dim=1)[1] == labels).sum()
                                total_val += labels.size(0)

                        print(
                            f'Val loss {val_loss / len(test_dataloader):.3f}. Val acc {correct_val / total_val * 100:.3f}.',
                            end=' ')
                        print(f'Took {time() - iter_time:.3f} seconds')
                        iter_time = time()

                        train_loss = running_loss / total_train
                        val_loss = val_loss / total_val

                        mlflow.log_metric('train_losses', train_loss, step=epoch)
                        mlflow.log_metric('val_loss', val_loss, step=epoch)

                        train_losses.append(train_loss)
                        val_losses.append(val_loss)

        return self.model, train_losses, val_losses

    def evaluate_model(self, test_subset, model, classes) -> EvaluationMetrics:
        y_test = []
        y_pred = []
        for img, label in test_subset:
            img = torch.Tensor(img)
            img = img.to(self.device)
            model.eval()
            prediction = model(img[None])

            final_pred = classes[torch.max(prediction, dim=1)[1]]

            # print(label, genre_dict[final_pred])

            y_test.append(label)
            y_pred.append(self.genre_dict[final_pred])

        precision, recall, fscore, support = score(np.array(y_test), np.array(y_pred), average='macro')
        return EvaluationMetrics(precision=precision,
                                 recall=recall,
                                 fscore=fscore)
