import math
from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def calc_shape_test(
    shape,
    out_channels,
    kernel_size=(5, 5),
    stride=(2, 2),
    dilation=(1, 1),
    padding=(0, 0),
):
    shape[-3] = out_channels
    # H_out and W_out
    shape[-2] = (
        shape[-2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
    ) // stride[0] + 1
    shape[-1] = (
        shape[-1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
    ) // stride[1] + 1
    return shape


class Net(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        # 1x28x28
        conv1_out_channel = 6
        conv2_out_channel = 6
        kernel_size_conv = (1, 1)
        stride_conv = (1, 1)
        maxpool_kernel = 2
        maxpool_stride = 2
        self.conv1 = nn.Conv2d(input_shape[0], conv1_out_channel, kernel_size_conv)
        x_shape = calc_shape_test(
            input_shape, conv1_out_channel, kernel_size_conv, stride_conv
        )

        self.pool = nn.MaxPool2d(maxpool_kernel, maxpool_kernel)

        self.conv2 = nn.Conv2d(conv1_out_channel, conv2_out_channel, kernel_size_conv)
        x_shape = calc_shape_test(
            x_shape, conv2_out_channel, kernel_size_conv, stride_conv
        )
        self.flatten = nn.Flatten()
        print(f"that shape{x_shape}")
        shape = math.prod(x_shape)

        self.fc1 = nn.Linear(shape, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        new_shape = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        # x = x.view(-1, new_shape)
        x = self.flatten(x)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def calc_shape(channel, prev_width, prev_height, kernel, stride):
    return (
        channel,
        calculate_2d_conv_pool_shape(prev_width, kernel, stride),
        calculate_2d_conv_pool_shape(prev_height, kernel, stride),
    )


def run_training():
    net = Net([1, 28, 28])

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    )

    batch_size = 4

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    sample, classes = next(iter(trainloader))

    net(sample)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # for epoch in range(1):  # loop over the dataset multiple times
    #
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data
    #
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #
    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:  # print every 2000 mini-batches
    #             print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
    #             running_loss = 0.0
    #
    # print("Finished Training")


def calculate_2d_conv_pool_shape(
    height_in, kernel_size, stride=1, dilation=1, padding=0
):
    return floor(
        ((height_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
    )


if __name__ == "__main__":
    # x = torch.randn([4, 1, 28, 28])
    # print(x.shape)
    # conv1 = Conv2d(1, 16, 5, 2)
    # x = conv1(x)
    # print(x.shape)
    # conv2 = Conv2d(16, 16, 5, 2)
    # x = conv2(x)
    # print(x.shape)
    # new_shape = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
    # x = x.view(-1, new_shape)
    # # x = x.view(-1, (x[1:].size()))
    # print(x.shape)
    #   flops_estimator = FlopsEstimator()
    x = torch.randn([4, 1, 28, 28])
    print(x.shape)
    # net = Net([1, 28, 28])
    # flops_estimator = FlopsEstimator()
    # flops_estimator.estimate_flops(net, x)
