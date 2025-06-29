import json
import logging
import logging.config
import os
from pathlib import Path
from matplotlib.font_manager import json_dump, json_load
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import nni

from torchvision.transforms import transforms
from elasticai.explorer.config import DeploymentConfig, HWNASConfig, ModelConfig
from elasticai.explorer.data_to_csv import build_search_space_measurements_file
from elasticai.explorer.explorer import Explorer
from elasticai.explorer.hw_nas.search_space.construct_sp import (
    CombinedSearchSpace,
    yml_to_dict,
)
from elasticai.explorer.knowledge_repository import (
    KnowledgeRepository,
    HWPlatform,
)
from elasticai.explorer.platforms.deployment.compiler import PicoCompiler
from elasticai.explorer.platforms.deployment.device_communication import (
    PicoHost,
)
from elasticai.explorer.platforms.deployment.manager import (
    CONTEXT_PATH,
    PicoHWManager,
    Metric,
)
from elasticai.explorer.platforms.generator.generator import (
    PicoGenerator,
)
from elasticai.explorer.trainer import MLPTrainer
from settings import ROOT_DIR

nni.enable_global_logging(False)
logging.config.fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger("explorer.main")


def setup_knowledge_repository_pico() -> KnowledgeRepository:
    knowledge_repository = KnowledgeRepository()
    knowledge_repository.register_hw_platform(
        HWPlatform(
            "pico",
            "Pico with RP2040 MCU and 2MB control memory",
            PicoGenerator,
            PicoHWManager,
            PicoHost,
            PicoCompiler,
        )
    )

    return knowledge_repository


def setup_mnist_for_cpp():

    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    mnist_test = datasets.MNIST(
        root="data/mnist", train=False, download=True, transform=transf
    )
    images = []
    labels = []

    for i in range(256):
        img, label = mnist_test[i]
        images.append(img.squeeze().numpy())
        labels.append(label)

    os.makedirs("data/cpp-mnist", exist_ok=True)
    with open("data/cpp-mnist/mnist_images.h", "w") as f:
        f.write("#ifndef MNIST_IMAGES_H\n#define MNIST_TEST_IMAGES_H\n\n")
        f.write("// 256 MNIST-Bilder (28x28), normal (0.0 - 1.0)\n")
        f.write("const float mnist_images[256][784] = {\n")

        for img in images:
            flat = img.flatten()
            f.write("  {\n    ")
            for i in range(784):
                f.write(f"{flat[i]:.6f}f")
                if i < 783:
                    f.write(", ")
                if (i + 1) % 16 == 0 and i != 783:
                    f.write("\n    ")
            f.write("\n  },\n")

        f.write("};\n\n#endif // MNIST_IMAGES_H\n")

    # Optional: Labels exportieren
    with open("data/cpp-mnist/mnist_labels.h", "w") as f:
        f.write("#ifndef MNIST_LABELS_H\n#define MNIST_LABELS_H\n\n")
        f.write("const int mnist_labels[256] = {\n  ")
        f.write(", ".join(str(l) for l in labels))
        f.write("\n};\n\n#endif // MNIST_LABELS_H\n")


def find_generate_measure_for_pico(
    explorer: Explorer,
    deploy_cfg: DeploymentConfig,
    hwnas_cfg: HWNASConfig,
    search_space: CombinedSearchSpace,
):
    explorer.choose_target_hw(deploy_cfg)
    explorer.generate_search_space(search_space)
    top_models = explorer.search(hwnas_cfg)

    # Creating Train and Test set from MNIST #TODO build a generic dataclass/datawrapper
    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    path_to_dataset = Path("data/mnist")
    trainloader: DataLoader = DataLoader(
        MNIST(path_to_dataset, download=True, transform=transf),
        batch_size=64,
        shuffle=True,
    )
    testloader: DataLoader = DataLoader(
        MNIST(path_to_dataset, download=True, train=False, transform=transf),
        batch_size=64,
    )

    setup_mnist_for_cpp()

    latency_measurements = []
    accuracy_measurements = []
    retrain_device = "cpu"
    for i, model in enumerate(top_models):
        mlp_trainer = MLPTrainer(
            device=retrain_device,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),  # type: ignore
        )
        mlp_trainer.train(model, trainloader=trainloader, epochs=3)
        mlp_trainer.test(model, testloader=testloader)
        model_name = "ts_model_" + str(i) + ".tflite"
        explorer.generate_for_hw_platform(model, model_name)
        explorer.hw_setup_on_target(Path("data/cpp-mnist"))

        try:
            latency = explorer.run_measurement(Metric.LATENCY, model_name)
        except:
            latency = json.loads("{ \"Latency\": { \"value\": -2, \"unit\": \"microseconds\"}}")

        try:
            accuracy = explorer.run_measurement(Metric.ACCURACY, model_name)
        except:

            accuracy = json.loads("{\"Accuracy\": { \"value\":  -2, \"unit\": \"percent\"}}")

        latency_measurements.append(latency)
        accuracy_measurements.append(accuracy)

    latencies = [latency["Latency"]["value"] for latency in latency_measurements]
    accuracies = [accuracy["Accuracy"]["value"] for accuracy in accuracy_measurements]
    df = build_search_space_measurements_file(
        latencies,
        accuracies,
        explorer.metric_dir / "metrics.json",
        explorer.model_dir / "models.json",
        explorer.experiment_dir / "experiment_data.csv",
    )
    logger.info("Models:\n %s", df)


if __name__ == "__main__":
    hwnas_cfg = HWNASConfig(config_path=Path("configs/pico/hwnas_config.yaml"))
    deploy_cfg = DeploymentConfig(
        config_path=Path("configs/pico/deployment_config.yaml")
    )
    model_cfg = ModelConfig(config_path=Path("configs/pico/model_config.yaml"))

    knowledge_repo = setup_knowledge_repository_pico()
    explorer = Explorer(knowledge_repo)
    explorer.set_model_cfg(model_cfg)
    search_space = yml_to_dict(
        Path("elasticai/explorer/hw_nas/search_space/search_space.yml")
    )
    search_space = CombinedSearchSpace(search_space)

    find_generate_measure_for_pico(explorer, deploy_cfg, hwnas_cfg, search_space) #type:ignore
