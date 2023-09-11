import torch
from torch import nn
from d2l import torch as d2l


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))
print(pool2d(X, (2, 2), 'avg'))

# 创建一个4x4的二维张量（tensor），数据类型为float32
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))

# 创建一个最大池化层，池化窗口大小为3x3，默认情况下，深度学习框架中的步幅与汇聚窗口的大小相同。
# 因此，如果我们使用形状为(3, 3)的池化窗口，那么默认情况下，我们得到的步幅形状为(3, 3)。
pool2d = nn.MaxPool2d(3)

# 使用最大池化层对输入张量X进行池化操作
# 最大池化层将3x3的窗口在输入张量上滑动，每次选取窗口内的最大值作为输出
# 所以输出张量的大小会变小
output = pool2d(X)
# 打印最大池化的结果
print(output)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))
