import torch  # 导入PyTorch库，用于深度学习任务
from torch import nn  # 导入PyTorch的神经网络模块
from d2l import torch as d2l  # 导入d2l库中的PyTorch工具，用于深度学习教育和研究


# 定义一个函数，用于创建VGG网络块
def vgg_block(num_convs, in_channels, out_channels):
    layers = []  # 创建一个空列表，用于存储神经网络层

    # 循环添加一系列卷积层，每个卷积层后面跟一个ReLU激活函数
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))  # 添加卷积层，指定输入通道数、输出通道数、卷积核大小和填充
        layers.append(nn.ReLU())  # 添加ReLU激活函数，用于引入非线性性
        in_channels = out_channels  # 更新输入通道数，以便下一个卷积层使用

    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # 添加最大池化层，用于下采样
    return nn.Sequential(*layers)  # 使用nn.Sequential组合所有层，并返回一个网络块
