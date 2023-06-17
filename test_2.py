from oodcls import OodCls
import torch
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
from torch.autograd import Variable

import torch.nn.functional as F

ood_cls = OodCls()

# 构建含有OOD类别的train_data

# 加载 CIFAR-10 数据集
# 加载过程中，调整 CIFAR-10 图像大小并转换为灰度图像
transform_cifar10 = transforms.Compose([
    transforms.Resize(28), 
    transforms.Grayscale(num_output_channels=1), 
    transforms.transforms.RandomEqualize(p=1),      #------直方图均衡化
    transforms.RandomAutocontrast(p=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_cifar10 = datasets.CIFAR10(root='./data/', train=True, download=True, transform=transform_cifar10)
test_cifar10 = datasets.CIFAR10(root='./data/', train=False, transform=transform_cifar10)

# 修改 CIFAR-10 的 label , test集和train集都要修改
label_list1 = train_cifar10.targets
label_list1 = [10 for label in label_list1]

label_list2 = test_cifar10.targets
label_list2 = [10 for label in label_list2]

train_cifar10.targets = torch.tensor(label_list1, dtype=torch.long)
test_cifar10.targets = torch.tensor(label_list2, dtype=torch.long)

train_cifar10.targets = train_cifar10.targets.tolist()
test_cifar10.targets = test_cifar10.targets.tolist()

# 加载 KMNIST 数据集

transform = transforms.Compose([
    transforms.Resize((28)),    # 缩放到相同的大小
    transforms.ToTensor(),         # 将图像转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化张量
])

train_kmnist = datasets.KMNIST(root='./data/', train=True, download=True, transform=transform)
test_kmnist = datasets.KMNIST(root='./data/', train=False, transform=transform)

label_list1 = train_kmnist.targets
label_list1 = [10 for label in label_list1]

label_list2 = test_kmnist.targets
label_list2 = [10 for label in label_list2]

# 将修改后的标签列表更新到数据集中
train_kmnist.targets = torch.tensor(label_list1, dtype=torch.long)
test_kmnist.targets = torch.tensor(label_list2, dtype=torch.long)
# print(test_cifar10)


# 检查数据类型是否与 mnist 一致
# print(type(train_cifar10.targets))

# 加载 fashionMNIST 数据集

train_fmnist = datasets.FashionMNIST('./data/', train=True, download=True, transform=transform)
test_fmnist = datasets.FashionMNIST('./data/', train=False, download=True, transform=transform)

label_list1 = train_fmnist.targets
label_list1 = [10 for label in label_list1]

label_list2 = test_fmnist.targets
label_list2 = [10 for label in label_list2]

train_fmnist.targets = torch.tensor(label_list1, dtype=torch.long)
test_fmnist.targets = torch.tensor(label_list2, dtype=torch.long)
# print(test_cifar10)

# 加载 MNIST 数据集

train_mnist = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
test_mnist = datasets.MNIST(root="./data/", train = False, transform = transform)


# 合并数据集

train_data = ConcatDataset([train_cifar10, train_mnist, train_kmnist, train_fmnist])
test_data = ConcatDataset([test_cifar10, test_mnist, test_kmnist, test_fmnist])


#***********************************************************************************************************
#***********************************************************************************************************

# 导入准备好的数据
imgs = torch.randn(torch.Size([100, 1, 28, 28]))
for i in range(100):
    imgs[i], _ = test_data[i*100]
# 加入cpu里
imgs = torch.tensor([item.cpu().detach().numpy() for item in imgs]).cuda()

# 导入随机生成的数据
# imgs = torch.randn(5, 1, 28, 28)
# imgs = torch.tensor([item.cpu().detach().numpy() for item in imgs]).cuda()


#**********************************************************************************************************
#**********************************************************************************************************


# 进行预测
preds = ood_cls.classify(imgs)

# 输出检查(在jupyter notebook里有输出，与预测值相符合)
img = torchvision.utils.make_grid(imgs)
img = img.cpu().numpy().transpose(1,2,0)

std = [0.5]
mean = [0.5]
img = img*std+mean
plt.imshow(img)





