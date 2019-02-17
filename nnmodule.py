# A single Layer
# torch.nn 专为深度学习设计的模块 核心数据结构：Module
# -- 既可以表示神经网络的某个层，也可表示一个包含很多层的神经网络

import torch as t
from torch import nn
from torch.autograd import Variable as V

class Linear(nn.Module): # 继承nn.Module
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()  #等价于nn.Module.__init__(self)
        self.w = nn.Parameter(t.randn(in_features, out_features))
        self.b = nn.Parameter(t.randn(out_features))

    def forward(self, x):
        x = x.mm(self.w)
        return x + self.b.expand_as(x)


# layer = Linear(8, 3)
# input = V(t.randn(2, 8))
# output = layer(input)  # 前向传播
# print(output)  # w*x+b
#
# for name, parameter in layer.named_parameters():
#     print(name, parameter)


# Multilayer Perceptron
class Perceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        # 不要忘记调用nn.Module的初始函数
        nn.Module.__init__(self)
        self.layer1 = Linear(in_features, hidden_features)
        self.layer2 = Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = t.sigmoid(x)
        return self.layer2(x)


perceptron = Perceptron(3, 5, 1)
input = V(t.randn(4, 3))
hidden = perceptron.layer1(input)
print(hidden)
output = perceptron.layer2(hidden)
print(output)


for name, parameter in perceptron.named_parameters():
    print(name, parameter)
