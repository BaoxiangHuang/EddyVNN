from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from torch import nn
import torch
from torch.nn.functional import softmax
from torchvision.models import resnet18

label_all = ['Alt purified AE', 'Alt purified CE', 'Outside eddy']

def conv1x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1x1x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride, padding=(0, 0, dilation), bias=False, groups=groups)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 1), stride=stride, bias=False)

class block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(block, self).__init__()
        self.conv1 = nn.Sequential(
            conv1x3(in_channels, out_channels, stride=stride),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv1x3(out_channels, out_channels)
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class My_CNN(nn.Module):
    def __init__(self, block, layers, num_classes=3, zero_init_residual=False):
        super(My_CNN, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(1, self.in_channels, kernel_size=(1, 1, 7), stride=(1, 1, 2), padding=(0, 0, 3))
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(1, 1, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(1, 1, 2))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(1, 1, 2))
        self.avgpool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion + 4,  num_classes)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x1 = x[:, :, :, :, :-4]
        x2 = x[:, :, :, :, -4:]
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)

        x1 = self.avgpool2(x1)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

def one_resnet18():
    return My_CNN(block, [2, 2, 2, 2])

def one_resnet34():
    return My_CNN(block, [3, 4, 6, 3])

def one_resnet50():
    return My_CNN(Bottleneck, [3, 4, 6, 3])

def one_resnet101():
    return My_CNN(Bottleneck, [3, 4, 23, 3])
