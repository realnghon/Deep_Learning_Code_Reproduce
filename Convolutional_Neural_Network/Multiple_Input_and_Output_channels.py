import torch
from d2l import torch as d2l


def corr2d_multi_in(X, K):
    # X是一个输入数据的列表，K是卷积核的列表
    # 该函数的目标是对每个通道的输入数据进行二维互相关操作，并将结果相加

    # 使用列表推导式遍历输入数据X和卷积核K，并计算每个通道的互相关结果
    # zip(X, K)将输入数据和卷积核按通道一一配对
    # d2l.corr2d(x, k)计算了输入数据x和卷积核k的二维互相关操作
    # sum(...) 将所有通道的结果相加
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


# 定义输入数据 X
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])

# 定义卷积核 K
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

# 使用 corr2d_multi_in 函数进行多通道的二维互相关操作
result = corr2d_multi_in(X, K)

# 打印结果
print(result)


def corr2d_multi_in_out(X, K):
    # X是输入数据，K是卷积核列表，每个卷积核都将产生一个通道的输出

    # 使用列表推导式迭代卷积核K，对每个卷积核执行corr2d_multi_in操作
    # corr2d_multi_in(X, k)计算每个卷积核k在输入数据X上的互相关结果
    # 结果会有多个通道，每个通道对应一个卷积核
    # torch.stack(...) 将所有通道的结果按照第0个维度（通道维度）叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


# 使用 torch.stack 将原始卷积核 K 与 K + 1 和 K + 2 叠加在一起
K = torch.stack((K, K + 1, K + 2), 0)

# 查看新卷积核的形状
print(K.shape)

print(corr2d_multi_in_out(X, K))

import torch  # 需要导入 torch 库


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape  # 获取输入特征图的通道数和尺寸信息
    c_o = K.shape[0]  # 获取输出通道数

    X = X.reshape((c_i, h * w))  # 将输入特征图变形为二维矩阵，形状为 (c_i, h * w)
    K = K.reshape((c_o, c_i))  # 将卷积核变形为二维矩阵，形状为 (c_o, c_i)

    # 执行全连接层中的矩阵乘法，相当于 1x1 卷积操作
    Y = torch.matmul(K, X)

    return Y.reshape((c_o, h, w))  # 将输出特征图还原为指定形状


# 创建一个随机输入特征图 X 和卷积核 K
X = torch.normal(0, 1, (3, 3, 3))  # 3通道的3x3x3输入特征图
K = torch.normal(0, 1, (2, 3, 1, 1))  # 2(输出通道数量)的3(卷积核数量，有多少个输入通道就有多少个卷积核)x1x1卷积核

# 使用定义的 corr2d_multi_in_out_1x1 函数执行 1x1 卷积操作
Y1 = corr2d_multi_in_out_1x1(X, K)

Y2 = corr2d_multi_in_out(X, K)

# 使用断言检查两种卷积操作的输出是否非常接近（误差小于 1e-6）
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
