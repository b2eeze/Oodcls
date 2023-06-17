
import torch
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
from torch.autograd import Variable

import torch.nn.functional as F


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

class OodCls:
    def __init__(self):
       self.M = Model()
       self.M.load_state_dict(torch.load('model_parameter_ood.pkl')) 
    
    def classify(self, imgs : torch.Tensor) -> torch.Tensor:
        
        # 获取输入张量的维度
        num_dims = imgs.ndim
        
        data_loader = torch.utils.data.DataLoader(dataset = imgs,
                                          batch_size = num_dims,
                                          shuffle = True)
        x = next(iter(data_loader))
        x = Variable(x)
        preds = self.M(x)
        _, preds = torch.max(preds, 1)
        
       
        
        return preds

ood_cls = OodCls()

# 准备输入数据
imgs = torch.randn(4, 1, 28, 28)

# 进行预测
preds = ood_cls.classify(imgs)

print("Predict Label is:", [i.item() for i in preds.data])



