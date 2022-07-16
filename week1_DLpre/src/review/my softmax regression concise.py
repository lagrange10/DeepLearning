from matplotlib.pyplot import show
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

def init_weights(m:nn.Linear) -> None:
    ''' 初始化线性层的权重 '''
    # m:nn.Module 神经网络的一层的基类
    if(nn.Linear == type(m)):
        nn.init.normal_(m.weight,std = 0.01)

net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))
net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(),lr = 0.1)

nums_epoch = 10
d2l.train_ch3(net,train_iter,test_iter,loss,nums_epoch,trainer)

show()

