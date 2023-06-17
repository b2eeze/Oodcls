from oodcls import OodCls
import torch

# 创建 OodCls 实例

ood_cls = OodCls()

# 准备输入数据
imgs = torch.randn(5, 1, 28, 28)
# 输入到GPU
imgs = torch.tensor([item.cpu().detach().numpy() for item in imgs]).cuda()

# 进行预测
preds = ood_cls.classify(imgs)
