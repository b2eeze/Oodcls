#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
from torch.autograd import Variable

import torch.nn.functional as F


# In[2]:


class Model(torch.nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        
        # 卷积层
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # 全连接层
        self.fc1 = torch.nn.Linear(128*7*7, 1024)
        self.fc2 = torch.nn.Linear(1024, 11)
        
    def forward(self, x):
        # 卷积层1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # 卷积层2，使用Dropout正则化
        x = F.relu(self.conv2(x))
        x = F.dropout(x, p=0.2)
        x = F.max_pool2d(x, 2)

        # 全连接层，使用L2正则化
        x = x.view(-1, 128*7*7)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        return x


# In[56]:


class OodCls:
    def __init__(self):
       self.M = Model()
       self.M = self.M.cuda()
       self.M.load_state_dict(torch.load('model_parameter_ood.pkl')) 
    
    def misc(self, x):
        # 转为灰度图像
        x_gray = transforms.Grayscale(x)
    
    def classify(self, imgs : torch.Tensor) -> torch.Tensor:
        
        # 获取输入张量的维度
        num_dims = imgs.size()[0]
        
        data_loader = torch.utils.data.DataLoader(dataset = imgs,
                                          batch_size = num_dims,
                                          shuffle = True)
        x = next(iter(data_loader))
        
        # 图形优化
        x_modify = self.misc(x)
        
        x_test = Variable(x)
        preds = self.M(x_test)
        _, preds = torch.max(preds, 1)
        print("Predict Label is:", [ i.item() for i in preds.data])
        
        
        img = torchvision.utils.make_grid(x)
        img = img.cpu().numpy().transpose(1,2,0)

        std = [0.5]
        mean = [0.5]
        img = img*std+mean
        plt.imshow(img)
        
        return preds.data


# In[57]:


ood_cls = OodCls()


# In[58]:


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


# In[59]:


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

# print(label_list)
# 将修改后的标签列表更新到数据集中
train_kmnist.targets = torch.tensor(label_list1, dtype=torch.long)
test_kmnist.targets = torch.tensor(label_list2, dtype=torch.long)
# print(test_cifar10)


# 检查数据类型是否与 mnist 一致
# print(type(train_cifar10.targets))


# In[60]:


# 加载 fashionMNIST 数据集

train_fmnist = datasets.FashionMNIST('./data/', train=True, download=True, transform=transform)
test_fmnist = datasets.FashionMNIST('./data/', train=False, download=True, transform=transform)

label_list1 = train_fmnist.targets
label_list1 = [10 for label in label_list1]

label_list2 = test_fmnist.targets
label_list2 = [10 for label in label_list2]

# print(label_list)
# 将修改后的标签列表更新到数据集中
train_fmnist.targets = torch.tensor(label_list1, dtype=torch.long)
test_fmnist.targets = torch.tensor(label_list2, dtype=torch.long)
# print(test_cifar10)

# 检查数据类型是否与 mnist 一致
# print(type(train_cifar10.targets))


# In[61]:


# 加载 MNIST 数据集

train_mnist = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
test_mnist = datasets.MNIST(root="./data/", train = False, transform = transform)

# 检查数据类型
# print(type(train_mnist.targets))


# In[62]:


# 合并数据集

train_data = ConcatDataset([train_cifar10, train_mnist, train_kmnist, train_fmnist])
test_data = ConcatDataset([test_cifar10, test_mnist, test_kmnist, test_fmnist])


# cifar_train = TensorDataset(train_cifar10.data.float(), train_cifar10.targets)
# 完成


# In[63]:


# print(test_data.target)


# In[87]:


# 导入准备好的数据

# imgs = [test_data[i][0] for i in range(3)]

imgs = torch.randn(torch.Size([100, 1, 28, 28]))
for i in range(100):
    imgs[i], _ = test_data[i*100]
    
# print(imgs)

imgs = torch.tensor([item.cpu().detach().numpy() for item in imgs]).cuda()



# In[88]:


print(imgs.shape)


# In[89]:


# 准备输入数据
# imgs = torch.randn(5, 1, 28, 28)
# imgs = torch.tensor([item.cpu().detach().numpy() for item in imgs]).cuda()

# 进行预测
preds = ood_cls.classify(imgs)

# print(preds.shape)

# print("Predict Label is:",  preds[2])


# In[ ]:




