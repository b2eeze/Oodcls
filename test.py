from oodcls import OodCls
import torch

# 创建 OodCls 实例
ood_classifier = OodCls()

# 准备输入数据
imgs = torch.randn(4, 1, 28, 28)

# 进行预测
preds = ood_classifier.classify(imgs)

print("Predict Label is:", [i.item() for i in preds.data])
