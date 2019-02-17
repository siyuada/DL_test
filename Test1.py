import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch as t
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
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


net = Net()
print(net)

params = list(net.parameters())
print(len(params))

# 查看有哪些参数
for name,parameters in net.named_parameters():
    print(name,":", parameters.size())

# nSamples*nChannels*Height*Width
input = Variable(t.randn(1, 1, 32, 32))
# 前向传播
# out = net(input)
# print(out.size())
# net.zero_grad()
# # 反向传播
# out.backward(Variable(t.ones(1, 10)))
#
# output = net(input)
target = Variable(t.arange(0, 10))
target = target.float()
# type()返回变量类型 .dtype返回数据类型
# print(output.dtype)
print(target.dtype)
# 损失函数
criterion = nn.MSELoss()
# loss = criterion(output, target)
# print(loss)
#
# net.zero_grad() # 将可学习参数的梯度清零
# print('Before conv1.bias grad')
# print(net.conv1.bias.grad)
# # 反向传播计算所有参数的梯度
# loss.backward()
# print('After backward conv1.bias grad')
# print(net.conv1.bias.grad)

# 使用优化方法更新网络权重和参数
# SGD
# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)
#
optimizer = optim.SGD(net.parameters(), lr=0.01)
# 首先梯度清零
optimizer.zero_grad()
# 计算loss
output = net(input)
loss = criterion(output, target)
# 反向传播
loss.backward()
# 参数更新
optimizer.step()
