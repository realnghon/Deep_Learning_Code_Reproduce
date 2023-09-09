import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# 定义真实的权重和偏置
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# 生成包含1000个样本的合成数据集
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 定义一个函数用于构造一个PyTorch数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器
    Args:
        data_arrays: 包含特征和标签数据的元组
        batch_size: 批处理大小
        is_train: 是否用于训练，默认为True
    Returns:
        数据迭代器
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# 定义Huber损失函数
def huber_loss(y_pred, y_true, delta=1.0):
    """计算Huber损失
    Args:
        y_pred: 模型的预测值
        y_true: 真实标签
        delta: Huber损失的阈值参数，默认为1.0
    Returns:
        计算得到的Huber损失
    """
    abs_diff = torch.abs(y_pred - y_true)
    # 使用torch.where函数进行逐元素比较和损失计算
    loss = torch.where(abs_diff < delta, 0.5 * abs_diff ** 2, delta * (abs_diff - 0.5 * delta))
    return loss.mean()


# 设置批处理大小
batch_size = 10
# 创建数据迭代器
data_iter = load_array((features, labels), batch_size)
# 获取数据迭代器的第一个批次数据
next(iter(data_iter))

# 创建一个包含一个线性层的神经网络
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
# 初始化权重参数为均值为0，标准差为0.01的正态分布，初始化偏置参数为0
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 创建一个随机梯度下降（SGD）优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练模型
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        # 计算Huber损失
        l = huber_loss(net(X), y, 1.0)
        # 梯度清零
        trainer.zero_grad()
        # 反向传播计算梯度
        l.backward()
        # 更新参数
        trainer.step()
    # 计算整个数据集上的损失
    l = huber_loss(net(features), labels, 1.0)
    print(f'epoch {epoch + 1}, loss {l:f}')

# 输出模型权重和偏置的估计误差
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
