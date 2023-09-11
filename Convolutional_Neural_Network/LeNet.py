import torch
from torch import nn
# 导入d2l库，d2l是一个用于深度学习教育的工具库
from d2l import torch as d2l

# 创建一个神经网络模型，这是一个Sequential模型，它按照顺序堆叠各层
net = nn.Sequential(
    # 添加一个卷积层，输入通道数为1，输出通道数为6，卷积核大小为5，填充为2，激活函数为ReLU
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
    # 添加最大池化层，池化核大小为2x2，步幅为2
    nn.MaxPool2d(kernel_size=2, stride=2),
    # 添加第二个卷积层，输入通道数为6，输出通道数为16，卷积核大小为5，激活函数为ReLU
    nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
    # 添加第二个最大池化层，池化核大小为2x2，步幅为2
    nn.MaxPool2d(kernel_size=2, stride=2),
    # 添加一个展平层，将多维的输入数据展平成一维
    nn.Flatten(),
    # 添加一个全连接层，输入特征数是16*5*5，输出特征数是120，激活函数为ReLU
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    # 添加一个全连接层，输入特征数是120，输出特征数是84，激活函数为ReLU
    nn.Linear(120, 84), nn.ReLU(),
    # 添加一个全连接层，输入特征数是84，输出特征数是10，这是最后一层，用于分类
    nn.Linear(84, 10))

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 设置批量大小为256
batch_size = 256
# 载入Fashion MNIST训练和测试数据集，每次迭代生成批量数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


def evaluate_accuracy_gpu(net, data_iter, device=None):  # 定义一个函数用于计算模型在数据集上的精度
    if isinstance(net, nn.Module):
        net.eval()  # 设置模型为评估模式，不进行梯度计算
        if not device:
            device = next(iter(net.parameters())).device  # 获取模型参数所在的设备
    # 创建一个累加器，用于存储正确预测的数量和总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():  # 在评估模式下，不进行梯度计算
        for X, y in data_iter:  # 遍历数据迭代器
            if isinstance(X, list):
                # 如果输入X是一个列表，将列表中的每个元素移动到指定设备上
                X = [x.to(device) for x in X]
            else:
                # 否则，将输入X移动到指定设备上
                X = X.to(device)
            y = y.to(device)  # 将标签y移动到指定设备上
            # 使用模型net对输入X进行预测，并计算精度
            metric.add(d2l.accuracy(net(X), y), y.numel())
    # 返回模型在数据集上的精度，即正确预测的数量除以总预测的数量
    return metric[0] / metric[1]


# @save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""

    # 定义一个函数init_weights，用于初始化模型的权重
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    # 应用初始化权重函数到模型net的每一层
    net.apply(init_weights)

    print('training on', device)
    net.to(device)  # 将模型移动到指定设备上
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 使用随机梯度下降优化器
    loss = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)

    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()  # 设置模型为训练模式
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()  # 清零梯度
            X, y = X.to(device), y.to(device)
            y_hat = net(X)  # 前向传播
            l = loss(y_hat, y)  # 计算损失
            l.backward()  # 反向传播
            optimizer.step()  # 更新参数
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

            # 打印训练过程中的损失和准确率
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))

        # 计算测试集上的准确率
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    # 打印最终的训练结果
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


lr, num_epochs = 0.1, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()
