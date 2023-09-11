import torch
from torch import nn


# 定义一个函数 comp_conv2d 用于计算卷积层的输出
def comp_conv2d(conv2d, X):
    # 将输入 X 调整为形状 (1, 1, height, width)，其中 1 表示批量大小和通道数都是 1
    X = X.reshape((1, 1) + X.shape)
    # 使用传入的卷积层 conv2d 对输入 X 进行卷积操作
    Y = conv2d(X)
    # 去除批量大小和通道数的维度，得到卷积结果的形状
    return Y.reshape(Y.shape[2:])


# 创建一个卷积层 nn.Conv2d，参数分别为输入通道数 1，输出通道数 1，卷积核大小为 3x3，padding 为 1
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
# 创建一个随机输入矩阵 X，大小为 (8, 8)
X = torch.rand(size=(8, 8))
# 调用 comp_conv2d 函数计算卷积结果的形状
output_shape = comp_conv2d(conv2d, X).shape

# 打印卷积结果的形状
print(output_shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
