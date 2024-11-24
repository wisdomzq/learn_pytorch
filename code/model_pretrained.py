#%%
import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
# 最后将图片分为1000类


train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)
# 分为10类

#对vgg16进行添加一层，是最后输出改为10类
vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))
print(vgg16_true)

#对vgg16进行修改，是最后输出改为10类
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)
#%% md
### 对现有网络模型进行使用及修改（以VGG16为例）
'''
pretrained为False时，加载的是一个随机初始化的模型，可以对其进行训练。
模型将使用随机初始化的权重。这意味着模型的权重没有经过预训练，需要从头开始进行训练。
在这种情况下，模型将不会具备捕捉通用图像特征的能力，而是需要根据特定任务的数据进行训练。

pretrained为True时，加载的是一个在ImageNet数据集上预训练的模型，可以对其进行微调。
这些预训练的权重经过了在大量图像上的训练，可以捕捉到通用的图像特征。
通过加载预训练权重，可以将VGG模型初始化为在ImageNet上训练得到的状态，并且这些权重可以作为初始参数用于特定任务的微调或迁移学习。
—
'''
