import json
import math

import nni
import torch
from nni.nas import strategy
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.experiment import NasExperiment
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from elasticai.explorer.cost_estimator import FlopsEstimator
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
    ##Parameter
    flops_weight = 0.5
    n_epochs = 3

    ##Cost-Estimation
    #flops as proxy metric for latency
    flops_estimator = FlopsEstimator(model_space= model)  
    flops = flops_estimator.estimate_flops_single_module()

    #set device to cpu to prevent memory error
    #TODO find workaround to use gpu on search but cpu on final retraining for deployment on pi
    device = "cpu"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(MNIST("data/mnist", download=True, transform=transf), batch_size=64, shuffle=True)
    test_loader = DataLoader(MNIST("data/mnist", download=True, train=False, transform=transf), batch_size=64)

    metric = {"default": 0, "accuracy" : 0, "flops log10": math.log10(flops)}
    for epoch in range(n_epochs):
        train_epoch(model, device, train_loader, optimizer, epoch)

        metric["accuracy"] = test_epoch(model, device, test_loader)
        
        metric["default"] = metric["accuracy"] - (metric["flops log10"] * flops_weight)
        nni.report_intermediate_result(metric)

    nni.report_final_result(metric)


   


def search():
    
    model_space = MLP()
    search_strategy = strategy.Random()
    evaluator = FunctionalEvaluator(evaluate_model)
    exp = NasExperiment(model_space, evaluator, search_strategy)
    exp.config.max_trial_number = 4
    exp.run(port=8081)
    top_models = exp.export_top_models(top_k=1, formatter="instance")
    for model_dict in exp.export_top_models(formatter='dict'):
        print(model_dict)
        with open("models/models.json", "w") as f:
            json.dump(model_dict, f)
    return top_models
