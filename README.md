# OodCls

`oodCls.py` 中定义了一个Oodcls类，封装了一个基于 PyTorch 的分类模型，对mnist数据集中的图像，可以正确识别所代表的数字(0-9)；同时，对于非数字的图像，识别为OOD类(10)。

## 使用方法

### 初始化模型

在使用 `OodCls` 进行分类预测之前，需要将训练好的模型加载到类中。可以使用以下代码进行模型初始化和加载。

```python
from oodcls import OodCls

ood_cls = OodCls()
```

### 进行预测

要使用 `OodCls` 进行分类预测，需要将输入数据传递给接口函数 `classify` 。该函数的参数是一个 `n*1*28*28` 的tensor（n是batch的大小，每个 `1*28*28` 的tensor表示的数字图像），输出是整数型n维tensor（n是batch的大小，每个整数在0~10范围内，代表分类结果）。

```python
# 进行预测

preds = ood_classifier.classify(imgs)

print("Predict Label is:", [i.item() for i in preds.data])

```

### 示例

在以下示例中，我们将使用 `OodCls` 类对一组随机数据进行分类预测。这里使用一个假的 `Model` 类作为模型。

```python
from oodcls import OodCls

# 初始化 OodCls 类
ood_cls = OodCls()

# 准备输入数据imgs
......

# 进行预测
preds = ood_classifier.classify(imgs)

print("Predict Label is:", [i.item() for i in preds.data])
```




