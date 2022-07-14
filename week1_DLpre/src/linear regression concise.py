import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):  
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays) #*是解包
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

from torch import ne, nn
#神经网络模块
#把输入为2d，输出1d的全连接层（每个输入都通过「矩阵-向量乘法」得到输出)
#Sequential是一个容器，它将输入送进第一层，第一层的输出传入第二层作为输入
net = nn.Sequential(nn.Linear(2, 1)) 

#初始化模型参数
net[0].weight.data.normal_(0,0.1) #结尾下划线是「替换方法」
net[0].bias.data.fill_(0)

#损失函数
loss = nn.MSELoss()

#优化算法
trainer = torch.optim.SGD(net.parameters(),lr = 0.03)

nums_epoch = 3
for epoch in range(nums_epoch):
    for X,y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features),labels)
    print(f'epoch {epoch + 1}, loss {l:f}')