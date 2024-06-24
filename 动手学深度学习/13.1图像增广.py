import torch
import torchvision
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
from matplotlib.image import imread

d2l.set_figsize()
img = d2l.Image.open('../data/img/cat2.jpg')
d2l.plt.imshow(img);
# plt.show()


def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
    plt.show()


shape_aug = torchvision.transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
# apply(img, shape_aug)
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, color_aug)

all_images = torchvision.datasets.CIFAR10(train=True, root="../data")
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8)
# plt.show()

# 我们还可以创建一个RandomColorJitter实例，并设置如何同时随机更改图像的亮度（brightness）
# 对比度（contrast）、饱和度（saturation）和色调（hue）。
apply(img, torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5))

# 多种组合
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)


# 图像增广训练
# 使用ToTensor实例将一批图像转换为深度学习框架所要求的格式，
# 即形状为（批量大小，通道数，高度，宽度）的32位浮点数，取值范围为0～1。
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])


def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader


def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)


def train_with_data_aug(train_augs, test_augs, net, epoch, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, epoch, devices)


# 训练参数
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)
epoch = 10

net.apply(init_weights)

# 不使用增广
train_with_data_aug(test_augs, test_augs, net,epoch)
# 训练加入增广，会减少过拟合
train_with_data_aug(train_augs, test_augs, net,epoch)