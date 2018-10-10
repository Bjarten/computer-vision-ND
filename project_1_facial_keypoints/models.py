## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

from collections import OrderedDict

class NamishNet(nn.Module):
    def __init__(self):
        super(NamishNet, self).__init__()

        self.conv1 = nn.Sequential(
            OrderedDict([
            ('conv1', nn.Conv2d(1, 32, kernel_size=5)),
            ('relu1', nn.ReLU())
        ])) # (32, 240, 240)

        self.maxp1 = nn.Sequential(OrderedDict([
            ('maxp1', nn.MaxPool2d(2, 2)),
            #('dropout1', nn.Dropout(0.1)),
            #('bachnorm1', nn.BatchNorm2d(32))
        ])) # (32, 240, 240)

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(32, 64, kernel_size=3)),
            ('relu2', nn.ReLU())]))

        self.maxp2 = nn.Sequential(OrderedDict([
            ('maxp2', nn.MaxPool2d(2, 2)),
            #('dropout2', nn.Dropout(0.2)),
            #('bachnorm2', nn.BatchNorm2d(64))
        ]))

        self.conv3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(64, 128, kernel_size=3)),
            ('relu3', nn.ReLU())]))

        self.maxp3 = nn.Sequential(OrderedDict([
            ('maxp3', nn.MaxPool2d(2, 2)),
            #('dropout4', nn.Dropout(0.3)),
            #('bachnorm3', nn.BatchNorm2d(128))
        ]))

        self.conv4 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(128, 256, kernel_size=2)),
            ('relu4', nn.ReLU())]))

        self.maxp4 = nn.Sequential(OrderedDict([
            ('maxp4', nn.MaxPool2d(2, 2)),
            #('dropout4', nn.Dropout(0.4)),
            #('bachnorm4', nn.BatchNorm2d(256))
        ]))

        self.fc1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(256 * 12 * 12, 1000)),
            ('relu5', nn.ReLU()),
            #('dropout5', nn.Dropout(0.5)),
            #('bachnorm4', nn.BatchNorm1d(1000))
        ]))

        self.fc2 = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(1000, 1000)),
            ('relu6', nn.ReLU()),
            #('dropout6', nn.Dropout(0.6)),
            #('bachnorm4', nn.BatchNorm1d(1000))
        ]))

        self.fc3 = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(1000, 136))]))

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxp1(out)
        out = self.conv2(out)
        out = self.maxp2(out)
        out = self.conv3(out)
        out = self.maxp3(out)
        out = self.conv4(out)
        out = self.maxp4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def __str__(self):
        pretty_net_str = ''
        for layer_name in self._modules:
            pretty_net_str += f'{layer_name}:\n'
            for items in getattr(self, layer_name):
                pretty_net_str += f'{items}\n'
            pretty_net_str += '\n'
        return pretty_net_str

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet_1, self).__init__()

        self.conv1 = nn.Sequential(
            OrderedDict([
            ('conv1', nn.Conv2d(1, 64, kernel_size=5, padding = 2)),
            ('relu1', nn.ReLU())]))

        self.maxp1 = nn.Sequential(OrderedDict([
            ('maxp1', nn.MaxPool2d(2, 2))]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(64, 128, kernel_size=5, padding = 2)),
            ('relu2', nn.ReLU())]))

        self.maxp2 = nn.Sequential(OrderedDict([
            ('maxp2', nn.MaxPool2d(2, 2))]))

        self.fc1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(128 *  56 *  56, 544)),
            ('relu3', nn.ReLU())]))

        self.fc2 = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(544, 272)),
            ('relu4', nn.ReLU())]))

        self.fc3 = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(272, 136))]))

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxp1(out)
        out = self.conv2(out)
        out = self.maxp2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

    def __str__(self):
        pretty_net_str = ''
        for layer_name in self._modules:
            pretty_net_str += f'{layer_name}:\n'
            for items in getattr(self, layer_name):
                pretty_net_str += f'{items}\n'
            pretty_net_str += '\n'
        return pretty_net_str
