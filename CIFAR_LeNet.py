#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# UTF-8编码格式！

import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch as t
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def run():
    t.multiprocessing.freeze_support()
    print('loop')
    show = ToPILImage()
    # 归一化
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    # 训练集
    trainset = tv.datasets.CIFAR10(root='\data', train=True, download=False, transform=transform)
    trainloader = t.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # 测试集
    testset = tv.datasets.CIFAR10(root='\data', train=False, download=False, transform=transform)
    testloader = t.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    (data, label) = trainset[100]
    print(classes[label])
    # im = transforms.ToPILImage()(((data+1)/2).float()).resize((100, 100))
    # im = show(((data+1)/2).float()).resize((100, 100))
    # show((data+1)/2).resize((100, 100))
    # im.show()
    # imshow(tv.utils.make_grid(data))

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    # imshow(tv.utils.make_grid(images))
    # for j in range(4):
    #     print('%5s' % classes[labels[j]])
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
    # im = show(tv.utils.make_grid((images+1)/2).float()).resize(400, 400)
    # im.show()

    net = Net()
    print(net)

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 训练网络
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 输入数据
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            # 梯度清零
            optimizer.zero_grad()

            # 前向后向传播
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 参数更新
            optimizer.step()

            # 打印log信息
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' \
                      % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # 测试
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    print('实际Label: ', ' '.join( \
        '%08s' % classes[labels[j]] for j in range(4)))

    # 计算网络预测的label
    outputs = net(Variable(images))
    _, predicted = t.max(outputs.data, 1)
    print('预测结果：', ' '.join('%5s' \
                             % classes[predicted[j]] for j in range(4)))

    # 在整个测试集上
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _,predicted = t.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        # 一个batch 4个数据

    print('10000张测试集中准确率为：%d %%' % (100 * correct/total))


if __name__ == '__main__':
    run()




