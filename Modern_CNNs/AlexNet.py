# 导入所需的Python库
import torch
from torch import nn
from d2l import torch as d2l

# 创建一个卷积神经网络（Convolutional Neural Network，CNN）模型
net = nn.Sequential(
    # 第一层卷积层：输入通道数为1，输出通道数为32，卷积核大小为5x5，填充为2，激活函数为ReLU
    nn.Conv2d(1, 32, kernel_size=5, padding=2), nn.ReLU(),
    # 第一层最大池化层：池化核大小为2x2，步幅为2
    nn.MaxPool2d(kernel_size=2, stride=2),
    # 第二层卷积层：输入通道数为32，输出通道数为64，卷积核大小为5x5，填充为2，激活函数为ReLU
    nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
    # 第二层最大池化层：池化核大小为2x2，步幅为2
    nn.MaxPool2d(kernel_size=2, stride=2),
    # 第三层卷积层：输入通道数为64，输出通道数为128，卷积核大小为3x3，填充为1，激活函数为ReLU
    nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
    # 第四层卷积层：输入通道数为128，输出通道数为128，卷积核大小为3x3，填充为1，激活函数为ReLU
    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
    # 第五层卷积层：输入通道数为128，输出通道数为64，卷积核大小为3x3，填充为1，激活函数为ReLU
    nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
    # 第三层最大池化层：池化核大小为2x2，步幅为2
    nn.MaxPool2d(kernel_size=2, stride=2),
    # 展平数据以供全连接层使用
    nn.Flatten(),
    # 第一个全连接层：输入特征数为64*3*3，输出特征数为512，激活函数为ReLU
    nn.Linear(64 * 3 * 3, 512), nn.ReLU(),
    # Dropout层，用于防止过拟合，丢弃概率为0.5
    nn.Dropout(p=0.5),
    # 第二个全连接层：输入特征数为512，输出特征数为256，激活函数为ReLU
    nn.Linear(512, 256), nn.ReLU(),
    # Dropout层，用于防止过拟合，丢弃概率为0.5
    nn.Dropout(p=0.5),
    # 第三个全连接层：输入特征数为256，输出特征数为10，用于分类
    nn.Linear(256, 10)
)

# 定义批量大小
batch_size = 256

# 加载Fashion MNIST数据集并分成训练集和测试集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义学习率和训练的总轮数
lr, num_epochs = 0.1, 10

# 使用GPU（如果可用）来训练模型，调用d2l库中的train_ch6函数
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

d2l.plt.show()
