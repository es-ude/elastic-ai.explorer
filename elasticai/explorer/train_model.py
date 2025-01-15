import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


def test(model):
    transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_loader = DataLoader(MNIST("data/mnist", download=True, train=False, transform=transf), batch_size=64)
    device = "cpu" #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        "\nTest set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct, len(test_loader.dataset), accuracy
        )
    )
    return accuracy


def train_epoch(
        model: torch.nn.Module, device, train_loader: DataLoader, optimizer, epoch
):
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
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def train(model: torch.nn.Module, epochs = 5):
    device = "cpu" #torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    transf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_loader = DataLoader(
        MNIST("data/mnist", download=True, transform=transf),
        batch_size=64,
        shuffle=True,
    )
    for epoch in range(epochs):
        train_epoch(model, device, train_loader, optimizer, epoch)
