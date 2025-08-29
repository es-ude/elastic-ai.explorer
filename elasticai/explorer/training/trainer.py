from abc import ABC, abstractmethod
import logging
from typing import Any, Tuple, Callable
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

        dataset = dataset_spec.dataset_type(
            dataset_spec.dataset_location,
            transform=dataset_spec.transform,
        )

        train_subset, test_subset, val_subset = random_split(
            dataset,
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
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
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


def accuracy_fn(output, target):
    pred = output.argmax(dim=1)
    correct = pred.eq(target).sum().item()
    return correct, target.size(0)


class SupervisedTrainer(Trainer):
    def __init__(
        self,
        device: str,
        optimizer: Optimizer,
        dataset_spec: DatasetSpecification,
        loss_fn: Any = nn.CrossEntropyLoss(),
        batch_size: int = 64,
        extra_metrics: dict[str, Callable] = {"accuracy": accuracy_fn},
    ):
        super().__init__(
            device,
            optimizer,
            dataset_spec,
            loss_fn,
            batch_size,
        )
        self.logger = logging.getLogger("explorer.RegressionTrainer")
        self.extra_metrics = extra_metrics

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

    def evaluate(
        self, model: nn.Module, data_loader: DataLoader, description="Validation"
    ):
        model.to(device=self.device)
        model.eval()
        total_loss = 0
        total = 0
        metric_totals = {name: 0.0 for name in self.extra_metrics}
        metric_counts = {name: 0 for name in self.extra_metrics}
        metric_avg = {name: 0.0 for name in self.extra_metrics}
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.loss_fn(output, target)
                total_loss += loss.item() * data.size(0)
                for name, metric_fn in self.extra_metrics.items():
                    val, count = metric_fn(output, target)
                    metric_totals[name] += val
                    metric_counts[name] += count
                total += target.size(0)
        avg_loss = total_loss / total

        for name, metric_total, metric_count in zip(
            metric_totals.keys(), metric_totals.values(), metric_counts.values()
        ):
            metric_avg[name] = metric_total / metric_count
            self.logger.info(
                f"{description} set: {name}: {metric_total}/{metric_count} ({metric_avg:.4f}%)"
            )
        self.logger.info(f"{description} set: Loss: {avg_loss:.4f}")
        return metric_avg, avg_loss

    def validate(self, model: nn.Module):
        return self.evaluate(model, self.val_loader, "Validation")

    def test(self, model: nn.Module):
        return self.evaluate(model, self.test_loader, "Test")


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

    def evaluate(self, model: nn.Module, data_loader: DataLoader, description):
        model.to(device=self.device)
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                reconstructed = model(batch)
                loss = self.loss_fn(reconstructed, batch)
                total_loss += loss.item()
        total_loss /= len(data_loader)
        self.logger.info("{} Loss: {:.6f}".format(description, total_loss))
        return None, total_loss

    def validate(self, model: nn.Module) -> Tuple[float | None, float]:
        return self.evaluate(model, self.val_loader, "Validation")

    def test(self, model: nn.Module) -> Tuple[float | None, float]:
        return self.evaluate(model, self.test_loader, "Test")
