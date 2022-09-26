import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

pretrained_net = torchvision.models.resnet18(pretrained=True)
# print(list(pretrained_net.children())[-3:])

"""拿到预训练层的前几层，children得到一个遍历一级子模型的生成器，做成list
以后去掉最后的全局平均池化和线性层，用来放上fcn【1x1卷积层+转置卷积层】
"""
net = nn.Sequential(*list(pretrained_net.children())[0:-2])
 
"""
对于cnn来说，输入图片的高宽是可以任意变的（从imagenet的224x224到VOC的320x480）
无论何种尺寸，这里的resnet总是将输入高宽变为1/32。
这里1x1卷积层直接放到21通道，是为了减少计算量，64大小的核计算很慢。
k = 2p+s 可以保证输出的宽高变为原来的s倍，这里希望变为32倍，得到像素级的语义分割。
"""
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))

def bilinear_kernel(in_channels, out_channels, kernel_size):
    """双线性插值"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

"""自定义的初始化方法"""
# conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
#                                 bias=False)
# conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))

W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)