## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

from collections import OrderedDict

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        self.layer1 = nn.Sequential(
            OrderedDict([
            ('conv1', nn.Conv2d(1, 32, kernel_size=5)),
            ('relu1',nn.ReLU()),
            ('bachnorm1', nn.BatchNorm2d(10))]))
        
        self.layer2 = nn.Sequential(OrderedDict([
            ('maxp1', nn.MaxPool2d(2,2))]))
        
        self.layer3 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(10, 10, kernel_size=3)),
            ('relu2', nn.ReLU())]))
        
        self.layer4 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(11*11*10,100 )),
            ('relu3', nn.ReLU()),
            ('dropout4',  nn.Dropout(0.4))]))
        
        self.layer5 = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(100, 50)),
            ('relu4', nn.ReLU()),
            ('dropout4',  nn.Dropout(0.4))]))
        
        self.layer6 = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(50, 10)),
            ('relu5', nn.ReLU())]))
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        # final output
        return F.log_softmax(out, dim=1)

    def __str__(self):
        pretty_net_str = ''
        for layer_name in self._modules:
            pretty_net_str += f'\n******** {layer_name} *********'
            for items in getattr(self, layer_name):
                pretty_net_str += f'\n{items}'
            pretty_net_str += '\n'
        return pretty_net_str
