# 网络模型的保存与读取
import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1 模型结构+模型参数
# 存在问题：加载模型时需要先定义模型结构
torch.save(vgg16, 'vgg16_method1.pth')
# 加载模型
model = torch.load('vgg16_method1.pth')
#

# 保存方式2 模型参数（官方推荐，空间更小）
torch.save(vgg16.state_dict(), 'vgg16_method2.pth') # 保存模型参数成字典

# 加载模型,从字典形式恢复成网络模型
vgg16.load_state_dict(torch.load('vgg16_method2.pth'))
print(vgg16)