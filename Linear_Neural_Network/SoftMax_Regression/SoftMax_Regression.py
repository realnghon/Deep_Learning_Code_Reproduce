import matplotlib.pyplot as plt  # 导入matplotlib库用于绘图
import torch  # 导入PyTorch库
from IPython import display  # 导入IPython的display模块，用于显示图像和动画
from d2l import torch as d2l  # 导入d2l.torch模块，其中包含一些有用的深度学习函数

batch_size = 256  # 定义批量大小
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 使用d2l库加载Fashion MNIST数据集并创建训练和测试数据迭代器

num_inputs = 784  # 定义输入特征数
num_outputs = 10  # 定义输出类别数

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)  # 初始化权重参数W，使用正态分布随机初始化
b = torch.zeros(num_outputs, requires_grad=True)  # 初始化偏置参数b


def softmax(X):
    X_exp = torch.exp(X)  # 计算输入的指数
    partition = X_exp.sum(1, keepdim=True)  # 计算分母，用于归一化
    return X_exp / partition  # 返回softmax归一化的结果


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)  # 定义一个简单的神经网络，包括矩阵乘法、加法和softmax操作


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])  # 计算交叉熵损失函数


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 如果y_hat是一个矩阵，则将其转换为向量
    cmp = y_hat.type(y.dtype) == y  # 比较预测值与真实标签是否相等
    return float(cmp.type(y.dtype).sum())  # 返回正确预测的数量


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 创建一个累加器来存储正确预测数和预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())  # 计算在数据集上的精度
    return metric[0] / metric[1]  # 返回精度


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_epoch_ch3(net, train_iter, loss, updater):  # 定义一个名为train_epoch_ch3的函数，接受四个参数（net, train_iter, loss, updater）
    # 函数的文档字符串，描述了函数的目标和用法
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()  # 如果net是PyTorch中的神经网络模型，则将其设置为训练模式

    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)  # 创建一个Accumulator对象，用于累积三个指标：训练损失、训练准确度、样本数

    for X, y in train_iter:  # 遍历训练数据迭代器中的每个批次（X是输入数据，y是标签）
        # 计算模型的预测值并计算损失
        y_hat = net(X)  # 使用模型net对输入数据X进行预测，得到预测值y_hat
        l = loss(y_hat, y)  # 计算预测值y_hat与实际标签y之间的损失

        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()  # 将优化器中的梯度清零
            l.mean().backward()  # 计算损失的均值并进行反向传播，更新模型参数
            updater.step()  # 使用优化器更新模型参数
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()  # 计算损失的总和并进行反向传播，更新模型参数
            updater(X.shape[0])  # 使用自定义的updater来更新模型参数，传入批次大小作为参数

        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())  # 累积训练损失、训练准确度和样本数

    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]  # 返回平均训练损失和平均训练精度


class Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 初始化Animator类的实例
        if legend is None:
            legend = []  # 如果没有传入图例信息，则初始化为空列表
        d2l.use_svg_display()  # 使用SVG格式显示图形（这里使用了d2l库的功能）
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)  # 创建图形和轴
        if nrows * ncols == 1:
            self.axes = [self.axes, ]  # 如果只有一个子图，将其包装成列表
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)  # 定义一个lambda函数，用于配置轴的属性
        self.X, self.Y, self.fmts = None, None, fmts  # 初始化X、Y、和fmts属性为None

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]  # 如果y不是可迭代对象，将其转换为包含一个元素的列表
        n = len(y)  # 获取y的长度
        if not hasattr(x, "__len__"):
            x = [x] * n  # 如果x不是可迭代对象，将其复制n次形成列表
        if not self.X:
            self.X = [[] for _ in range(n)]  # 如果X属性为None，初始化为包含n个空列表的列表
        if not self.Y:
            self.Y = [[] for _ in range(n)]  # 如果Y属性为None，初始化为包含n个空列表的列表
        for i, (a, b) in enumerate(zip(x, y)):  # 遍历x和y的元素对
            if a is not None and b is not None:
                self.X[i].append(a)  # 向第i个列表中添加x的值
                self.Y[i].append(b)  # 向第i个列表中添加y的值
        self.axes[0].cla()  # 清空子图的内容
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)  # 绘制图形
        self.config_axes()  # 配置轴属性
        display.display(self.fig)  # 显示图形
        display.clear_output(wait=True)  # 清除图形并等待下一次更新


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（定义见第3章）"""
    # 创建用于绘制动画的Animator对象
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):  # 遍历训练周期的次数
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)  # 调用train_epoch_ch3函数进行一个训练周期的训练
        test_acc = evaluate_accuracy(net, test_iter)  # 在测试集上评估模型的准确度
        # 将训练损失、训练准确度和测试准确度添加到动画中
        animator.add(epoch + 1, train_metrics + (test_acc,))

    train_loss, train_acc = train_metrics  # 获取最后一个训练周期的损失和准确度
    # 断言，用于检查训练损失和准确度是否满足某些条件
    assert train_loss < 0.5, train_loss  # 断言训练损失小于0.5
    assert train_acc <= 1 and train_acc > 0.7, train_acc  # 断言训练准确度在0.7到1之间
    assert test_acc <= 1 and test_acc > 0.7, test_acc  # 断言测试准确度在0.7到1之间
    plt.show()  # 显示绘制的图形


lr = 0.1  # 学习率


# 定义一个用于更新模型参数的updater函数，使用随机梯度下降（SGD）优化器
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


num_epochs = 10  # 训练周期的数量

# 调用train_ch3函数来训练模型
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
