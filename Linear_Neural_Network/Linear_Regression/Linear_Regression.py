# 导入所需的库
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# 定义真实的模型参数
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# 使用d2l库生成合成数据集，包括特征(features)和标签(labels)
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 定义一个函数来构造PyTorch数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 设置批量大小
batch_size = 10

# 创建一个数据迭代器，用于训练模型
data_iter = load_array((features, labels), batch_size)

# 获取迭代器的一个批次数据
next(iter(data_iter))

# 导入PyTorch的神经网络模块
from torch import nn

# 定义一个单层神经网络，包括线性层和激活函数
net = nn.Sequential(nn.Linear(2, 1))

# 初始化网络权重和偏差
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义均方误差损失函数
loss = nn.MSELoss()

# 定义随机梯度下降(SGD)优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 定义训练轮数
num_epochs = 3

# 迭代训练模型
for epoch in range(num_epochs):
    for X, y in data_iter:
        # 计算模型的预测值
        l = loss(net(X), y)

        # 梯度清零
        trainer.zero_grad()

        # 反向传播求梯度
        l.backward()

        # 更新模型参数
        trainer.step()

    # 计算整个训练集上的损失
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# 获取神经网络中第一个层（唯一一个层）训练后的权重参数
w = net[0].weight.data

# 计算权重估计误差
print('w的估计误差：', true_w - w.reshape(true_w.shape))

# 获取训练后的偏差参数
b = net[0].bias.data

# 计算偏差估计误差
print('b的估计误差：', true_b - b)
