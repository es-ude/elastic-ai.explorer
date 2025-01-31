from abc import ABC, abstractmethod
import logging
from matplotlib import transforms
import torch
from torch import nn, optim
from torch.utils.data import DataLoader



class Trainer(ABC):
    @abstractmethod
    def train(self, model: nn.Module, trainloader: DataLoader, epochs: int):
        pass
    
    @abstractmethod
    def test(self, model: nn.Module, testloader: DataLoader) -> float:
        pass

    @abstractmethod
    def train_epoch(self, model: nn.Module, trainloader: DataLoader, epoch: int):
        pass

class MLPTrainer(Trainer):
    def __init__(self, device: str,
                  optimizer: optim, loss_fn = nn.CrossEntropyLoss()):

        self.logger = logging.getLogger("explorer.MLPTrainer")
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, model: nn.Module, trainloader: DataLoader, epochs: int):
        """Train the given model on the traindata

        Args:
            model (nn.Module): model
            trainloader (DataLoader): Data
            epochs (int): epochs
        """
        
        for epoch in range(epochs):
            self.train_epoch(model=model, trainloader=trainloader, epoch=epoch)

        
    def test(self, model: nn.Module, testloader: DataLoader) -> float:
        """Test the model on the 

        Args:
            testloader (DataLoader): The data for testing.

        Returns:
            float: Accuracy on test data.
        """

        test_loss = 0
        correct = 0
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(testloader.dataset)
        accuracy = 100.0 * correct / len(testloader.dataset)
        self.logger.info(
            "Test set: Accuracy: {}/{} ({:.0f}%)\n".format(
                correct, len(testloader.dataset), accuracy
            )
        )
        return accuracy

    def train_epoch(self, model: nn.Module, trainloader: DataLoader, epoch: int):
        """train for an epoch using

        Args:
            epoch (int): Current epoch
        """
        
        model.train(True)
        for batch_idx, (data, target) in enumerate(trainloader):
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
                        len(trainloader.dataset),
                        100.0 * batch_idx / len(trainloader),
                        loss.item(),
                    )
                )

        