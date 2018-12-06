## define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from torchvision import models

from collections import OrderedDict

# *** Conv2d output dimensions ***
# height_out = (height_in + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1
# width_out = (width_in + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1
# weights_out = height_out * width_out * channels_out
#
# With values: strid = 1, padding = 0, dilation = 1
# height_out = height_in - kernel_size + 1
# width_out = width_in - kernel_size + 1
#
# *** MaxPool2d output dimensions ***
# height_out = (height_in + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1
# width_out = (width_in + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1
# weights_out = height_out * width_out * channels_out
#
# With values: strid = 2, padding = 0, dilation = 1
# height_out = (height_in - kernel_size)/2 + 1
# width_out = (width_in - kernel_size)/2 + 1

class NaimishNet(nn.Module):
    def __init__(self, image_size, output_size = 136, kernels = [5,5,5,5],out_channels = [32,64,128,256],
                dropout_p = [0, 0, 0, 0, 0, 0], use_padding=True, use_maxp = True):
        super(NaimishNet, self).__init__() 
        # padding only support odd numbered kernels in this implementation
        self.use_padding = use_padding
        
        # init padding
        if self.use_padding:
            self.padding = [int((k-1)/2) for k in kernels]
        else:
            self.padding = [0,0,0,0]
            
        # Find the size of the last maxp output. 
        last_maxp_size = image_size
        for idx, val in enumerate(kernels):
            if self.use_padding:
                last_maxp_size = last_maxp_size//2
            else:
                last_maxp_size = (last_maxp_size - (val-1))//2
        last_maxp_size = out_channels[3] * last_maxp_size * last_maxp_size

        self.conv1 = nn.Sequential(
            OrderedDict([
            ('conv1', nn.Conv2d(1, out_channels[0], kernel_size=kernels[0], padding=self.padding[0])),
            ('relu1', nn.ReLU())
            ])) # (32, 252, 252)                        
        
        if use_maxp:
            self.maxp1 = nn.Sequential(OrderedDict([
                ('maxp1', nn.MaxPool2d(2, 2)),
                ('dropout1', nn.Dropout(dropout_p[0])),
                ('bachnorm1', nn.BatchNorm2d(out_channels[0]))
                ])) # (32, 126, 126)
        else:
            self.maxp1 = nn.Sequential(OrderedDict([
                ('maxp1', nn.AvgPool2d(2, 2)),
                ('dropout1', nn.Dropout(dropout_p[0])),
                ('bachnorm1', nn.BatchNorm2d(out_channels[0]))
                ])) # (32, 126, 126)

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(out_channels[0], out_channels[1], kernel_size=kernels[1], padding=self.padding[1])),
            ('relu2', nn.ReLU())
            ])) # (64, 122, 122)
        
        if use_maxp:
            self.maxp2 = nn.Sequential(OrderedDict([
                ('maxp2', nn.MaxPool2d(2, 2)),
                ('dropout2', nn.Dropout(dropout_p[1])),
                ('bachnorm2', nn.BatchNorm2d(out_channels[1]))
                ])) # (64, 61, 61)
        else:
            self.maxp2 = nn.Sequential(OrderedDict([
                ('maxp2', nn.AvgPool2d(2, 2)),
                ('dropout2', nn.Dropout(dropout_p[1])),
                ('bachnorm2', nn.BatchNorm2d(out_channels[1]))
                ])) # (64, 61, 61)
            
        self.conv3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(out_channels[1], out_channels[2], kernel_size=kernels[2], padding=self.padding[2])),
            ('relu3', nn.ReLU())
            ])) # (128, 59, 59)

        if use_maxp:
            self.maxp3 = nn.Sequential(OrderedDict([
                ('maxp3', nn.MaxPool2d(2, 2)),
                ('dropout3', nn.Dropout(dropout_p[2])),
                ('bachnorm3', nn.BatchNorm2d(out_channels[2]))
                ])) # (128, 29, 29)
        else:
            self.maxp3 = nn.Sequential(OrderedDict([
                ('maxp3', nn.AvgPool2d(2, 2)),
                ('dropout3', nn.Dropout(dropout_p[2])),
                ('bachnorm3', nn.BatchNorm2d(out_channels[2]))
                ])) # (128, 29, 29)
            
        self.conv4 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(out_channels[2], out_channels[3], kernel_size=kernels[3], padding=self.padding[3])),
            ('relu4', nn.ReLU())
            ])) # (256, 27, 27)
        
        if use_maxp:
            self.maxp4 = nn.Sequential(OrderedDict([
                ('maxp4', nn.MaxPool2d(2, 2)),
                ('dropout4', nn.Dropout(dropout_p[3])),
                ('bachnorm4', nn.BatchNorm2d(out_channels[3]))
                ]))  # (256, 13, 13)
        else:
            self.maxp4 = nn.Sequential(OrderedDict([
                ('maxp4', nn.AvgPool2d(2, 2)),
                ('dropout4', nn.Dropout(dropout_p[3])),
                ('bachnorm4', nn.BatchNorm2d(out_channels[3]))
                ]))  # (256, 13, 13)
        
        self.fc1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(last_maxp_size, 1024)),
            ('relu5', nn.ReLU()),
            ('dropout5', nn.Dropout(dropout_p[4])),
            ('bachnorm5', nn.BatchNorm1d(1024))
            ])) # (36864, 1024)

        self.fc2 = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(1024, 1024)),
            ('relu6', nn.ReLU()),
            ('dropout6', nn.Dropout(dropout_p[5])),
            ('bachnorm6', nn.BatchNorm1d(1024))
            ])) # (1024, 1024)

        self.fc3 = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(1024, output_size))
            ])) # (1024, 136)

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
        super(LeNet, self).__init__()

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
    
# Transfer learning

class vgg11_conv5_1(nn.Module):
    def __init__(self):
        super(vgg11_conv5_1, self).__init__()
        vgg11 = models.vgg11(pretrained=True).features
        # freeze training for all layers
        for param in vgg11.parameters():
            param.requires_grad_(False)
                    
        modules = list(vgg11.children())[:-3] 
        
        self.features = nn.Sequential(*modules)
       
        self.keypoints_estimator =  nn.Sequential(
            nn.Linear(100352, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 136))
                       
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.keypoints_estimator(x)
        return x
    
class vgg11_maxp4(nn.Module):
    def __init__(self):
        super(vgg11_maxp4, self).__init__()
        vgg11 = models.vgg11(pretrained=True).features
        # freeze training for all layers
        for param in vgg11.parameters():
            param.requires_grad_(False)
                    
        modules = list(vgg11.children())[:-5] 
        
        self.features = nn.Sequential(*modules)
       
        self.keypoints_estimator =  nn.Sequential(
            nn.Linear(100352, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 136))
                        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.keypoints_estimator(x)
        return x
    
class vgg11_conv4_2(nn.Module):
    def __init__(self):
        super(vgg11_conv4_2, self).__init__()
        vgg11 = models.vgg11(pretrained=True).features
        # freeze training for all layers
        for param in vgg11.parameters():
            param.requires_grad_(False)
                    
        modules = list(vgg11.children())[:-6] 
        
        self.features = nn.Sequential(*modules)
       
        self.keypoints_estimator =  nn.Sequential(
            nn.Linear(401408, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 136))
                        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.keypoints_estimator(x)
        return x
    
class vgg11_conv4_1(nn.Module):
    def __init__(self):
        super(vgg11_conv4_1, self).__init__()
        vgg11 = models.vgg11(pretrained=True).features
        # freeze training for all layers
        for param in vgg11.parameters():
            param.requires_grad_(False)
                    
        modules = list(vgg11.children())[:-8] 
        
        self.features = nn.Sequential(*modules)
       
        self.keypoints_estimator =  nn.Sequential(
            nn.Linear(401408, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 136))
                        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.keypoints_estimator(x)
        return x
    
class vgg11_maxp3(nn.Module):
    def __init__(self):
        super(vgg11_maxp3, self).__init__()
        vgg11 = models.vgg11(pretrained=True).features
        # freeze training for all layers
        for param in vgg11.parameters():
            param.requires_grad_(False)
                    
        modules = list(vgg11.children())[:-10] 
        
        self.features = nn.Sequential(*modules)
       
        self.keypoints_estimator =  nn.Sequential(
            nn.Linear(200704, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 136))
                        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.keypoints_estimator(x)
        return x
    
class vgg11_maxp2(nn.Module):
    def __init__(self):
        super(vgg11_maxp2, self).__init__()
        vgg11 = models.vgg11(pretrained=True).features
        # freeze training for all layers
        for param in vgg11.parameters():
            param.requires_grad_(False)
                    
        modules = list(vgg11.children())[:-15] 
        
        self.features = nn.Sequential(*modules)
       
        self.keypoints_estimator =  nn.Sequential(
            nn.Linear(401408, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 136))
                        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.keypoints_estimator(x)
        return x
    
class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        n_inputs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(n_inputs, 136)
                        
    def forward(self, x):
        x = self.resnet18(x)
        return x
    
class resnet18_grayscale(nn.Module):
    def __init__(self):
        super(resnet18_grayscale, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        # change from supporting color to gray scale images
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        n_inputs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(n_inputs, 136)
                        
    def forward(self, x):
        x = self.resnet18(x)
        return x
    
class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        n_inputs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(n_inputs, 136)
                        
    def forward(self, x):
        x = self.resnet18(x)
        return x
    
class resnet50(nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        n_inputs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(n_inputs, 136)
                        
    def forward(self, x):
        x = self.resnet50(x)
        return x
    
class resnet50_grayscale(nn.Module):
    def __init__(self):
        super(resnet50_grayscale, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        # change from supporting color to gray scale images
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        n_inputs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(n_inputs, 136)
                        
    def forward(self, x):
        x = self.resnet50(x)
        return x
    
class squeezenet_grayscale(nn.Module):
    def __init__(self):
        super(squeezenet_grayscale, self).__init__()

        squeezenet1_1 = models.squeezenet1_1(pretrained=True).features
        modules = list(squeezenet1_1.children()) 
   
        # change from supporting color to gray scale images
        modules[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2))
        
        self.features = nn.Sequential(*modules)
        self.classifier =  nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, 136, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=13, stride=1, padding=0))
                       
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), 136)