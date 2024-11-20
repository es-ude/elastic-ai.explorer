import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from elasticai.explorer import explorer
from elasticai.generator.generator import PIGenerator

from nni.nas.profiler.pytorch.flops import FlopsProfiler

def test(model, test_loader):
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset), accuracy))
    return accuracy

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

def train(model: torch.nn.Module):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(MNIST("data/mnist", download=True, transform=transf), batch_size=64, shuffle=True)
    for epoch in range(5):
        train_epoch(model, device, train_loader, optimizer, epoch)


#Try to estimate flops of the given model
def estimate_flops(model, dataloader):

    sample, target= next(iter(dataloader))
    profiler = FlopsProfiler(model, sample)

    print("Flops: ", profiler.expression)


    

if __name__ == '__main__':
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    top_models=explorer.search()
    test_loader = DataLoader(MNIST("data/mnist", download=True, train=False, transform=transf), batch_size=64)
    for i, top_model in enumerate(top_models):
        
        estimate_flops(top_model, test_loader)
        train(top_model)
        test(top_model, test_loader)
        generator= PIGenerator()

        if not os.path.isdir("models/ts_models"):
            os.mkdir("models/ts_models") 
        ts_model= generator.generate(top_model, path="models/ts_models/model " +str(i))
        test(ts_model, test_loader)
        data, target= next(iter(test_loader))
        torch.testing.assert_close(top_model(data), ts_model(data))
