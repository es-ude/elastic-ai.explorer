from abc import ABC, abstractmethod
import logging

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, random_split

from elasticai.explorer.data import DatasetInfo
from sklearn.model_selection import train_test_split


class Trainer(ABC):

    @abstractmethod
    def __init__(
        self,
        device: str,
        optimizer: Optimizer,
        dataset_info: DatasetInfo,
        loss_fn: _Loss = nn.CrossEntropyLoss(),
        batch_size: int = 64,
    ):
        pass

    @abstractmethod
    def train(self, model: nn.Module, epochs: int):
        pass

    @abstractmethod
    def validate(self, model: nn.Module) -> float:
        pass

    @abstractmethod
    def test(self, model: nn.Module) -> float:
        pass

    @abstractmethod
    def train_epoch(self, model: nn.Module, epoch: int):
        pass


class MLPTrainer(Trainer):
    """Trainer class for MLPs written in Pytorch."""

    def __init__(
        self,
        device: str,
        optimizer: Optimizer,
        dataset_info: DatasetInfo,
        loss_fn=nn.CrossEntropyLoss(),
        batch_size: int = 64,
    ):

        self.logger = logging.getLogger("explorer.MLPTrainer")
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        train_dataset = dataset_info.dataset_type(
            dataset_info.dataset_location,
            train=True,
            download=True,
            transform=dataset_info.transform,
        )
        test_dataset = dataset_info.dataset_type(
            dataset_info.dataset_location,
            train=True,
            download=True,
            transform=dataset_info.transform,
        )

        train_subset, val_subset = random_split(
            train_dataset,
            [0.9, 0.1],
            generator=torch.Generator().manual_seed(1),
        )

        self.train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    def train(self, model: nn.Module, epochs: int):
        """
        Args:
            model: Model to train.
            epochs: Number of epochs.
        """

        for epoch in range(epochs):
            self.train_epoch(model=model, epoch=epoch)

    def validate(self, model: nn.Module):
        """
        Args:
            model: The NN-Model to test.

        Returns:
            float: Accuracy on test data.
        """

        test_loss = 0
        correct = 0
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.val_loader.dataset)
        accuracy = 100.0 * correct / len(self.val_loader.dataset)
        self.logger.info(
            "Validation set: Accuracy: {}/{} ({:.0f}%)\n".format(
                correct, len(self.val_loader.dataset), accuracy
            )
        )
        return accuracy

    def test(self, model: nn.Module) -> float:
        """
        Args:
            model: The NN-Model to test.

        Returns:
            float: Accuracy on test data.
        """

        test_loss = 0
        correct = 0
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        accuracy = 100.0 * correct / len(self.test_loader.dataset)
        self.logger.info(
            "Test set: Accuracy: {}/{} ({:.0f}%)\n".format(
                correct, len(self.test_loader.dataset), accuracy
            )
        )
        return accuracy

    def train_epoch(self, model: nn.Module, epoch: int):
        """Trains model for only one epoch.

        Args:
            model: The NN-Model to test.
            epoch: Current epoch number.
        """
        model.to(device=self.device)
        model.train(True)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                self.logger.debug(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        loss.item(),
                    )
                )
