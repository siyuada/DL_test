#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# UTF-8编码格式！

import torch as t
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    """
    define repeat module
    """

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),  # 填充为1
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        print('x', x.size())
        out = self.left(x)
        print(out.size())
        if self.right is None:
            residual = x
            print('w', residual.size())
        else:
            residual = self.right(x)
            print('b', residual.size())

        # residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    """
    Main module: Resnet34
    contains many layers, every layer contains residual layer
    _make_layer function to construct different layer

    num_class -- using in the Linear layer
    """
    def __init__(self, num_class=1000):
        super(ResNet, self).__init__()
        # The first layer
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)  # kernel size, stride, padding
        )

        def _make_layer(inchannel, outchannel, block_num, stride=1):
            """
            Structure of every layer
            """
            shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            # the first block need above shortcut because of the dismatch in channel number

            layer = []
            layer.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

            for i in range(1, block_num):
                layer.append(ResidualBlock(outchannel, outchannel))

            return nn.Sequential(*layer)

        self.layer1 = _make_layer(64, 64, 3, stride=2)
        self.layer2 = _make_layer(64, 128, 4, stride=2)
        self.layer3 = _make_layer(128, 256, 6, stride=2)
        self.layer4 = _make_layer(256, 512, 3, stride=2)

        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.pre(x)
        print(x.size())
        x = self.layer1(x)
        print(x.size())
        x = self.layer2(x)
        print(x.size())
        x = self.layer3(x)
        print(x.size())
        x = self.layer4(x)
        print(x.size())
        x = F.avg_pool2d(x, 7)  # input, kernel size
        print(x.size())
        x = x.view(x.size(0), -1)
        print(x.size())
        return self.fc(x)


model = ResNet()
input = t.autograd.Variable(t.randn(1, 3, 224, 224))
# for name, parameter in model.named_parameters():
#     print(name, parameter.size())
print(model)
out = model(input)
print(out.size())  # 仅仅为输入通过该网络之后的输出

from torchvision import models
resnet34 = models.resnet34(pretrained=True, num_classes=1000)
out = resnet34(input)
print(out.size())