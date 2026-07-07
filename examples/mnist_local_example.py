import argparse
import logging
import logging.config
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import torch
from torch import nn, optim
from torch.utils.data import Subset
from torchvision import transforms

from elasticai.explorer.explorer import Explorer
from elasticai.explorer.hw_nas.estimators import ParamEstimator, TrainMetricsEstimator
from elasticai.explorer.hw_nas.hw_nas import HWNASParameters, SearchStrategy
from elasticai.explorer.hw_nas.optimization_criteria import OptimizationCriteria
from elasticai.explorer.knowledge_repository import KnowledgeRepository
from elasticai.explorer.training.data import DatasetSpecification, MNISTWrapper
from elasticai.explorer.training.trainer import SupervisedTrainer, accuracy_fn
from settings import ROOT_DIR

logger = logging.getLogger("explorer.main")


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_dataset_spec(data_dir: Path, limit: int, seed: int) -> DatasetSpecification:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = MNISTWrapper(root=data_dir, transform=transform, download=True)
    if limit > 0:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
        dataset = Subset(dataset, indices)

    return DatasetSpecification(
        dataset=dataset,
        deployable_dataset_path=data_dir,
        shuffle=True,
        split_seed=seed,
    )


def build_criteria(
    dataset_spec: DatasetSpecification,
    device: str,
    batch_size: int,
    estimation_epochs: int,
) -> OptimizationCriteria:
    trainer = SupervisedTrainer(
        device,
        dataset_spec,
        batch_size=batch_size,
        extra_metrics={"accuracy": accuracy_fn},
    )
    criteria = OptimizationCriteria()
    criteria.register_objective(
        estimator=TrainMetricsEstimator(
            trainer,
            metric_name="loss",
            n_estimation_epochs=estimation_epochs,
        ),
        weight=-1.0,
    )
    criteria.register_objective(
        estimator=ParamEstimator(),
        transform=lambda params: params / 1_000_000,
        weight=-0.05,
    )
    return criteria


def run_mnist_search(args: argparse.Namespace) -> None:
    logging.config.fileConfig(
        ROOT_DIR / "logging.conf",
        disable_existing_loggers=False,
    )
    device = choose_device(args.device)
    data_dir = ROOT_DIR / args.data_dir
    search_space = ROOT_DIR / args.search_space
    experiment_name = args.experiment_name

    dataset_spec = build_dataset_spec(data_dir, args.limit, args.seed)
    criteria = build_criteria(
        dataset_spec,
        device,
        args.batch_size,
        args.estimation_epochs,
    )

    explorer = Explorer(KnowledgeRepository(), experiment_name=experiment_name)
    explorer.generate_search_space(search_space)
    top_models = explorer.search(
        search_strategy=SearchStrategy.RANDOM_SEARCH,
        optimization_criteria=criteria,
        hw_nas_parameters=HWNASParameters(
            max_search_trials=args.trials,
            top_n_models=args.top_models,
            count_only_completed_trials=True,
        ),
    )
    if not top_models:
        raise RuntimeError("No valid model was found in the search space.")

    model = top_models[0]
    trainer = SupervisedTrainer(
        device,
        dataset_spec,
        loss_fn=nn.CrossEntropyLoss(),
        batch_size=args.batch_size,
        extra_metrics={"accuracy": accuracy_fn},
    )
    trainer.configure_optimizer(optim.Adam(model.parameters(), lr=args.lr))
    trainer.train(model, epochs=args.retrain_epochs, early_stopping=False)
    test_metrics, test_loss = trainer.test(model)

    explorer.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = explorer.model_dir / "mnist_local_model.pt"
    torch.save(model, model_path)

    logger.info("Best model:\n%s", model)
    logger.info("Test loss: %.4f", test_loss)
    logger.info("Test accuracy: %.4f", test_metrics["accuracy"])
    logger.info("Saved model to %s", model_path)
    logger.info("Experiment output: %s", explorer.experiment_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local MNIST HW-NAS search and retrain the best model."
    )
    parser.add_argument("--trials", type=int, default=10) # 2
    parser.add_argument("--top-models", type=int, default=1)
    parser.add_argument("--estimation-epochs", type=int, default=1)
    parser.add_argument("--retrain-epochs", type=int, default=5) # 1
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=0)  # 2000
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument("--experiment-name", default="mnist_local")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mnist"))
    parser.add_argument(
        "--search-space",
        type=Path,
        default=Path("examples/search_space_examples/mnist_local_search_space.yaml"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_mnist_search(parse_args())
