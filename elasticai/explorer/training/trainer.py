from abc import ABC, abstractmethod
import logging
from typing import Any, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim.optimizer import Optimizer
from elasticai.explorer.training.data import DatasetSpecification


class Trainer(ABC):

    def __init__(
        self,
        device: str,
        optimizer: Optimizer,
        dataset_spec: DatasetSpecification,
        loss_fn: Any = nn.CrossEntropyLoss(),
        batch_size: int = 64,
    ):
        self.logger = logging.getLogger("explorer.Trainer")
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        train_subset, test_subset, val_subset = random_split(
            dataset_spec.dataset,
            dataset_spec.test_train_val_ratio,
            generator=torch.Generator().manual_seed(dataset_spec.split_seed),
        )

        self.train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=dataset_spec.shuffle
        )
        self.val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=dataset_spec.shuffle
        )
        self.test_loader = DataLoader(
            test_subset, batch_size=batch_size, shuffle=dataset_spec.shuffle
        )

    def train(
        self,
        model: nn.Module,
        epochs: int,
        early_stopping: bool = True,
        patience: int = 3,
        min_delta: float = 0.0,
    ):
        """Override this method to customize behavior."""

        self.epochs = epochs

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = model.state_dict()

        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            self.train_epoch(model=model, epoch=epoch)
            _, val_loss = self.validate(model=model)
            if val_loss == None:
                raise ValueError("Trainer.validate() does not return Validation Loss")

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{patience}"
                )

            if early_stopping and patience_counter >= patience:
                self.logger.info("Early stopping triggered.")
                break

        if early_stopping:
            model.load_state_dict(best_model_state)
            self.logger.info("Loaded best model from early stopping.")

    @abstractmethod
    def validate(self, model: nn.Module) -> Tuple[float | None, float]:
        """
        Returns:
            Tuple[float, float]: (Accuracy, Loss)
        """
        pass

    @abstractmethod
    def test(self, model: nn.Module) -> Tuple[float | None, float]:
        """
        Returns:
            Tuple[float, float]: (Accuracy, Loss)
        """
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
        dataset_spec: DatasetSpecification,
        loss_fn: Any = nn.CrossEntropyLoss(),
        batch_size: int = 64,
    ):
        super().__init__(
            device,
            optimizer,
            dataset_spec,
            loss_fn,
            batch_size,
        )
        self.logger = logging.getLogger("explorer.MLPTrainer")

    def validate(self, model: nn.Module) -> Tuple[float | None, float]:
        """
        Args:
            model: The NN-Model to validate.

        Returns:
            float: Accuracy on validation data.
        """

        val_loss = 0.0
        correct = 0
        total = 0

        model.to(self.device)
        model.eval()

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.loss_fn(output, target)
                val_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        val_loss /= total
        accuracy = 100.0 * correct / total

        self.logger.info(
            f"Validation set: Accuracy: {correct}/{total} ({accuracy:.2f}%)"
        )
        self.logger.info(f"Validation set: Loss: {val_loss:.4f}")
        return accuracy, val_loss

    def test(self, model: nn.Module) -> Tuple[float | None, float]:
        """
        Args:
            model: The NN-Model to test.

        Returns:
            float: Accuracy on test data.
        """

        test_loss = 0
        correct = 0
        total = 0
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.loss_fn(output, target)
                test_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        test_loss /= total
        accuracy = 100.0 * correct / total
        self.logger.info(
            "Test set: Accuracy: {}/{} ({:.0f}%)\n".format(correct, total, accuracy)
        )
        self.logger.info(f"Test set: Loss: {test_loss:.4f}")
        return accuracy, test_loss

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
                        len(self.train_loader.dataset),  # type: ignore
                        100.0 * batch_idx / len(self.train_loader),
                        loss.item(),
                    )
                )


class ReconstructionAutoencoderTrainer(Trainer):

    def __init__(
        self,
        device: str,
        optimizer: Optimizer,
        dataset_spec: DatasetSpecification,
        loss_fn: Any = nn.MSELoss(),
        batch_size: int = 64,
    ):
        super().__init__(
            device,
            optimizer,
            dataset_spec,
            loss_fn,
            batch_size,
        )
        self.logger = logging.getLogger("explorer.AutoencoderTrainer")

    def train_epoch(self, model: nn.Module, epoch: int):
        model.to(device=self.device)
        model.train()
        train_loss = 0

        for data in self.train_loader:
            data = data.to(torch.float32)
            data = data.to(self.device)
            self.optimizer.zero_grad()
            reconstructed = model(data)
            loss = self.loss_fn(reconstructed, data)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        train_loss /= len(self.train_loader)
        self.logger.debug("Train Epoch: {}\tLoss: {:.6f}".format(epoch, train_loss))

    def validate(self, model: nn.Module) -> Tuple[float | None, float]:
        model.to(device=self.device)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                reconstructed = model(batch)
                loss = self.loss_fn(reconstructed, batch)
                val_loss += loss.item()
        val_loss /= len(self.val_loader)

        self.logger.info("Test Loss: {:.6f}".format(val_loss))
        return None, val_loss

    def test(self, model: nn.Module) -> Tuple[float | None, float]:
        model.to(device=self.device)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                reconstructed = model(batch)
                loss = self.loss_fn(reconstructed, batch)
                test_loss += loss.item()
        test_loss /= len(self.test_loader)
        self.logger.info("Test Loss: {:.6f}".format(test_loss))
        return None, test_loss
