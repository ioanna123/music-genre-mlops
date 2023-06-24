import os
from abc import ABC
from time import time

import torch
from torch import nn
from tqdm import tqdm

from genre_classification.data_model.criterion import Criterion
from genre_classification.trainer.optimizer import Optimizer, OptimizerBase
from settings import LR


class TLModelBase(ABC):

    def __init__(self,
                 model,
                 n_classes: int = 10,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 criterion=Criterion,
                 optimizer=Optimizer
                 ):
        self.model = model
        self.n_classes = n_classes
        self.device = device
        self.criterion = criterion
        self.optimizer_option = optimizer

        # Fix the trainable parameters
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Number of Input Features in the Last Fully Connected Layer
        in_features = self.model.fc.in_features

        # Replacing the Last Fully Connected Layer
        fc = nn.Linear(in_features=in_features, out_features=self.n_classes)
        self.model.fc = fc
        #
        # Updating the Weights and Bias of the last layer
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

        # Define the Loss and Optimizer Functions
        # self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = torch.optim.Adam(params_to_update, lr=0.001)

        self.optimizer = OptimizerBase(params_to_update, LR).optimizer(self.optimizer_option)

    # def train_setup

    @staticmethod
    def _save_model(checkpoint_path: str, model, correct_val, total_val):
        torch.save(model, os.path.join(checkpoint_path, f'checkpoint_{correct_val / total_val * 100:.2f}'))

    def train(self, test_dataloader, train_dataloader, num_epoch: int = 10):
        steps = 0
        train_losses, val_losses = [], []
        device = self.device

        self.model.to(device)
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
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Logging
                if steps % 10 == 0:
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

                    train_losses.append(running_loss / total_train)
                    val_losses.append(val_loss / total_val)
            self._save_model(checkpoint_path='checkpoints', model=self.model, correct_val=correct_val,
                             total_val=total_val)
        return self.model, train_losses, val_losses
