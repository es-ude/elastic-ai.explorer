import nni
import torch
import torch.nn as nn
from nni.nas import strategy
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.experiment import NasExperiment
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import json

from elasticai.generator.generator import PIGenerator
from elasticai.explorer.search_space import MLP

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


def search():
    model_space = MLP()
    search_strategy = strategy.Random()
    evaluator = FunctionalEvaluator(evaluate_model)
    exp = NasExperiment(model_space, evaluator, search_strategy)
    exp.config.max_trial_number = 3
    exp.run(port=8081)
    top_models = exp.export_top_models(top_k=1, formatter="instance")
    for model_dict in exp.export_top_models(formatter='dict'):
        print(model_dict)
        with open("models/models.json", "w") as f:
            json.dump(model_dict, f)
    return top_models


