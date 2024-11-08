import torch
import torch.nn as nn
import torch.nn.functional as F
import nni
from nni.nas.nn.pytorch import LayerChoice, ModelSpace, MutableDropout, MutableLinear


class MLP(ModelSpace):  # should inherit ModelSpace rather than nn.Module
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(self.conv2(x), 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = self.fc2(self.dropout2(F.relu(self.fc1(x))))
        output = F.log_softmax(x, dim=1)
        return output

    import nni
    import torch
    import torch.nn as nn
    from nni.nas import strategy
    from nni.nas.evaluator import FunctionalEvaluator
    from nni.nas.experiment import NasExperiment
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import MNIST

    from nni_test.search_space import MLP

    def train_epoch(model: torch.nn.Module, device, train_loader: DataLoader, optimizer, epoch):
        loss_fn = nn.CrossEntropyLoss()
        model.train(True)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    def test_epoch(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset), accuracy))
        return accuracy

    def evaluate_model(model: torch.nn.Module):
        global accuracy
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_loader = DataLoader(MNIST("data/mnist", download=True, transform=transf), batch_size=64, shuffle=True)
        test_loader = DataLoader(MNIST("data/mnist", download=True, train=False, transform=transf), batch_size=64)

        for epoch in range(3):
            train_epoch(model, device, train_loader, optimizer, epoch)
            accuracy = test_epoch(model, device, test_loader)
            nni.report_intermediate_result(accuracy)

        nni.report_final_result(accuracy)

    if __name__ == '__main__':
        model_space = MLP()
        search_strategy = strategy.Random()
        evaluator = FunctionalEvaluator(evaluate_model)
        exp = NasExperiment(model_space, evaluator, search_strategy)
        exp.config.max_trial_number = 3
        exp.run(port=8081)

        for model_dict in exp.export_top_models(formatter='dict'):
            print(model_dict)
            with open("models.json", "w") as f:
                json.dump(model_dict, f)