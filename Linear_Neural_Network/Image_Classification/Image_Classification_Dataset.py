import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

# 使用SVG格式显示图形（前提是d2l库支持SVG显示）
d2l.use_svg_display()

# 创建一个数据转换（transform）对象，将图像数据从PIL类型转换为32位浮点数格式，
# 并将像素值除以255，以确保所有像素的数值在0到1之间
trans = transforms.ToTensor()

# 下载并加载FashionMNIST数据集的训练集和测试集
# root参数指定数据集存储的根目录，如果不存在将会自动下载
# train=True表示加载训练集，train=False表示加载测试集
# transform参数将数据集中的图像应用上述定义的数据转换
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

# 打印训练集和测试集的样本数量
len(mnist_train), len(mnist_test)

# 查看训练集中第一个样本的图像数据的形状
mnist_train[0][0].shape


def get_fashion_mnist_labels(labels):
    # 定义Fashion-MNIST数据集中对应标签的文本标签列表
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    # 使用列表推导式将数值标签转换为文本标签并返回
    # labels参数是一个包含数值标签的列表
    # int(i)将数值标签转换为整数，然后从text_labels列表中获取对应的文本标签
    # 最终返回一个包含文本标签的列表，与输入的数值标签一一对应
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  # @save
    # 计算绘图区域的大小（以英寸为单位）
    figsize = (num_cols * scale, num_rows * scale)
    # 创建一个包含多个子图（axes）的图形窗口
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    # 将多个子图压缩成一个一维数组，以便逐个处理
    axes = axes.flatten()
    # 遍历图像列表以及对应的子图
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 如果图像是PyTorch张量，则将其转换为NumPy数组并绘制
            ax.imshow(img.numpy())
        else:
            # 如果图像是PIL图片，则直接绘制
            ax.imshow(img)
        # 隐藏子图的x轴和y轴刻度
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        # 如果提供了标题信息，则设置子图的标题
        if titles:
            ax.set_title(titles[i])
    # 返回绘制的子图列表
    return axes


# 从训练数据集中获取一个包含18个样本的小批量数据（batch）
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))

# 将获取的图像数据重新形状为(18, 28, 28)的三维张量，并显示在2行9列的子图中
# 使用get_fashion_mnist_labels函数为每个子图添加对应的文本标签
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

# 显示图像子图
d2l.plt.show()

# 定义批量大小为256的数据加载器（data loader）
batch_size = 256


# 定义获取数据加载器工作进程数量的函数
# 这里返回4表示使用4个进程来读取数据，以提高数据加载效率
def get_dataloader_workers():  # @save
    """使用4个进程来读取数据"""
    return 4


def load_data_fashion_mnist(batch_size, resize=None):  # @save
    # 定义数据变换（transform）操作的列表，初始为将图像转换为张量
    trans = [transforms.ToTensor()]
    # 如果指定了resize参数，将在数据变换前插入一个将图像大小调整为resize的操作
    if resize:
        trans.insert(0, transforms.Resize(resize))
    # 创建数据变换操作的组合
    trans = transforms.Compose(trans)
    # 下载并加载Fashion-MNIST训练集和测试集
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)

    # 创建训练集和测试集的数据加载器，并指定批量大小、是否打乱数据以及数据加载的工作进程数量
    train_loader = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                   num_workers=get_dataloader_workers())
    test_loader = data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers())

    # 返回训练集和测试集的数据加载器
    return train_loader, test_loader


# 使用load_data_fashion_mnist函数加载数据，批量大小为32，图像大小调整为64x64
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)

# 遍历训练集的一个批量数据并打印其形状和数据类型
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
