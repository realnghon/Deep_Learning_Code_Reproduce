import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def huber_loss(y_pred, y_true, delta=1.0):
    abs_diff = torch.abs(y_pred - y_true)
    # abs_diff 是一个张量，而不是标量，因此，不能像标量一样进行比较操作 <
    # 应该使用逐元素比较。可以使用 torch.where函数来处理这个问题
    loss = torch.where(abs_diff < delta, 0.5 * abs_diff ** 2, delta * (abs_diff - 0.5 * delta))
    return loss.mean()


batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))

from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = huber_loss(net(X), y, 1.0)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = huber_loss(net(features), labels, 1.0)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
