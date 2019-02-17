import torch as t
from torch.autograd import Variable as V
from matplotlib import pyplot as plt
from IPython import display
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

# 设置随机数种子
t.manual_seed(1000)

def get_fake_data(batch_size = 8):
    x = t.rand(batch_size, 1) * 20  # 8行1列 *20
    y = x * 2 + (1 + t.randn(batch_size, 1))*3
    # rand 正态分布 randn 标准分布
    return x, y

# x, y = get_fake_data(20)
# # 散点图
# plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())
# #plt.plot(x.numpy(), y.numpy())
# plt.show()

lr = 0.001
w = V(t.rand(1, 1), requires_grad=True)
b = V(t.zeros(1, 1), requires_grad=True)
for ii in range(2000):
    x, y = get_fake_data()

    # 计算loss
    y_pred = x.mm(w) + b.expand_as(y)
    print(y_pred)
    loss = 0.5 * (y_pred - y) ** 2
    loss = loss.sum()

    # backward
    # dloss = 1
    # dy_pred = dloss * (y_pred - y)
    #
    # dw = x.t().mm(dy_pred)
    # db = dy_pred.sum()
    #
    # # update
    # w.sub_(lr * dw)
    # b.sub_(lr * db)
    # print(w, b)

    # 自动反向传播
    loss.backward()

    # 更新参数
    w.data.sub_(lr * w.grad.data)
    b.data.sub_(lr * b.grad.data)

    # 梯度清零
    w.grad.data.zero_()
    b.grad.data.zero_()

    if ii % 1000 == 0:  # 画图
        display.clear_output(wait=True)
        x = t.arange(0., 20).view(-1, 1)
        y = x.mm(w.data) + b.data.expand_as(x)
        plt.plot(x.numpy(), y.numpy())  # predicted

        x2, y2 = get_fake_data(batch_size=20)
        plt.scatter(x2.numpy(), y2.numpy())

        plt.xlim(0, 20)
        plt.ylim(0, 41)
        plt.show()
        plt.pause(1)

print(w.data.squeeze(), b.data.squeeze())