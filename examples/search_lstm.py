import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader


class SineDataset(Dataset):
    def __init__(self, seq_length=10, total_samples=1000):
        x = np.linspace(0, 100, total_samples + seq_length)
        noise_level = 0.1
        self.data = (
            np.sin(x)
            + 0.5 * np.sin(3 * x)
            + 0.2 * np.sin(7 * x)
            + np.random.normal(scale=noise_level, size=x.shape)
        )
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.data[idx : idx + self.seq_length]
        target = self.data[idx + self.seq_length]
        return torch.tensor(seq, dtype=torch.float32).unsqueeze(-1), torch.tensor(
            target, dtype=torch.float32
        )


class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.linear(last_output)


def train(model, train_loader):
    # 0.028953095546891225
    # 0.013809413783353647
    # 0.011882943366034198
    num_epochs = 100
    learning_rate = 0.01
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for seqs, targets in train_loader:
            outputs = model(seqs)
            loss = criterion(outputs.squeeze(), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f}"
        )


def validate(model, test_loader):
    model.eval()
    preds = []
    targets = []
    total_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for seqs, target in test_loader:
            target = target.unsqueeze(1)
            output = model(seqs)
            preds.append(output.item())
            targets.append(target.item())
            loss = criterion(output, target)
            total_loss += loss.item()
    print(total_loss / len(test_loader))
    # Plot
    plt.plot(targets, label="True")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.title("Sine Wave Prediction")
    plt.show()


# class TestAutoencoderTrainer:
#     def setup_method(self):
#         self.dataset_spec = DatasetSpecification(DatasetExample, Path(""), None)
#         self.autoencoder = SimpleLSTMAutoencoder(INPUT_DIM)
#
#     def test_autoencoder_trainer(self):
#
#         autoencoder_trainer = ReconstructionAutoencoderTrainer(
#             "cpu",
#             optimizer=torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3),  # type: ignore
#             dataset_spec=self.dataset_spec,
#         )
#         autoencoder_trainer.train(self.autoencoder, 20)
#         assert autoencoder_trainer.validate(self.autoencoder)[1] > 0
#         assert autoencoder_trainer.test(self.autoencoder)[1] > 0
#
#
# def search_for_timeseries(explorer: Explorer, hwnas_cfg: HWNASConfig, search_space):
#     deploy_cfg = DeploymentConfig(config_path=Path("configs/deployment_config.yaml"))
#     explorer.choose_target_hw(deploy_cfg)
#     explorer.generate_search_space(search_space)
#     dataset_spec = DatasetSpecification(DatasetExample, Path(""), None)
#
#     top_models = explorer.search(hwnas_cfg, dataset_spec, MLPTrainer)
#
#     retrain_device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#     for i, model in enumerate(top_models):
#         print(f"found model {i}:  {model}")
#
#         mlp_trainer = MLPTrainer(
#             device=retrain_device,
#             optimizer=Adam(model.parameters(), lr=1e-3),
#             dataset_spec=dataset_spec,
#         )
#         mlp_trainer.train(model, epochs=3)
#         mlp_trainer.test(model)
#         print("=================================================")
#         model_name = "ts_model_" + str(i) + ".pt"
#
#         explorer.generate_for_hw_platform(model, model_name)


if __name__ == "__main__":
    seq_length = 50
    batch_size = 32
    dataset = SineDataset(seq_length=seq_length)
    train_size = int(0.8 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    model = SimpleLSTM()
    train(model, train_loader)
    validate(model, test_loader)
    # hwnas_cfg = HWNASConfig(config_path=Path("configs/hwnas_config.yaml"))
    # deploy_cfg = DeploymentConfig(config_path=Path("configs/deployment_config.yaml"))
    # knowledge_repo = setup_knowledge_repository_pi()
    # explorer = Explorer(knowledge_repo)
    #
    # search_space = Path("examples/search_space_example.yaml")
    #
    # search_for_timeseries(explorer, hwnas_cfg, search_space)
