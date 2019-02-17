from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
import torch as t
from torch import nn
from torch.autograd import Variable as V
import numpy as np

print("gaibiangaibianforgithub")

to_tensor = ToTensor()  # img -> tensor
to_pil = ToPILImage
lena = Image.open('lenat.bmp').convert('L')  # .convert('L') 转换成灰度图
# lena.show()  # 用电脑自带照片查看
# plt.figure("Original lena")
# plt.imshow(lena)
# plt.show()

input = to_tensor(lena).unsqueeze(0)  # 在第几维上增加一维

# 锐化卷积层
# kernel = t.ones(3, 3)/-9

kernel[1][1] = 1
conv = nn.Conv2d(1, 1, (3, 3), 1, bias=False)
conv.weight.data = kernel.view(1, 1, 3, 3)  # 改变卷积核的参数
print(kernel)

pool = nn.AvgPool2d(3, 3)
print(list(pool.parameters()))

out1 = pool(V(input))
pool_l = ToPILImage()(out1.data.squeeze(0))

out = conv(V(input))
conv_l = ToPILImage()(out.data.squeeze(0))  # 将tensor转回PILImage
print(type(kernel))
# plt.figure("Conv lena")
# plt.imshow(pool_l)
# plt.title("Conv lena")
# plt.axis("off")
# plt.show()

input = V(t.randn(2, 3))
linear = nn.Linear(3, 4)
h = linear(input)
print(h)

bn = nn.BatchNorm1d(4)
bn.weight.data = t.ones(4) * 2  # 设置标准差为2
bn.bias.data = t.zeros(4)
print(list(bn.named_parameters()))

bn_out = bn(h)
#  输出均值 和 方差
print(bn_out.mean(0), bn_out.var(0, unbiased=False))  # unbiased_计算时分母不-1 1/n

# 每个元素以0.5的概率丢弃
dropout = nn.Dropout(0.5)
print(bn_out)
o = dropout(bn_out)
print(o)  # 有一半左右的数变为0

# 激活函数 ReLU(x)=max(0,x)
relu = nn.ReLU(inplace=True)
print(input)
output = relu(input)
print(output)  # 小于0的变为0
print(input)  # 设置了inplace为true，所以input也变化 false不变

## 前馈传播网络的简单写法！
# 用Sequential
# 1
# conv2d(input(输入通道), output(输出通道)即有多少个卷积核，每个卷积核输出一个结果, size(kernel.size))
net1 = nn.Sequential()
net1.add_module('conv', nn.Conv2d(3, 3, 3))
net1.add_module('batchnorm', nn.BatchNorm2d(3))
net1.add_module('activation_layer', nn.ReLU())

# 2
net2 = nn.Sequential(nn.Conv2d(3, 3, 3), nn.BatchNorm2d(3), nn.ReLU())
#nn.Conv2d(3, 3, 3),
# 3
from collections import OrderedDict
net3 = nn.Sequential(OrderedDict([('conv', nn.Conv2d(3, 3, 3)),
                                  ('bn', nn.BatchNorm2d(3)),
                                  ('relu', nn.ReLU())]))
print("net1: ", net1)
print("net2: ", net2)
print("net3: ", net3)

t.manual_seed(2)
input = V(t.randn(1, 3, 4, 4))
print(input)
print("net1: ", net1(input).data)
print("net2: ", net2(input).data)
print("net3: ", net3(input).data)

# print("net1: ", (net1.batchnorm(input)), net1.activation_layer(net1.batchnorm(input)))
# print("net2: ", (net2[0](input)), net2[1](net2[0](input)))
# print("net3: ", (net3.bn(input)), net3.relu(net3.bn(input)))