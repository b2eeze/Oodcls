
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
        
        return preds





