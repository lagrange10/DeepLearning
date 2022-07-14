from re import L
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2,-3.4])
true_b = 4.2
example_nums = 1000
features, labels = d2l.synthetic_data(true_w, true_b, example_nums)

def load_array(data_arrays, batch_size, is_tarin = True):
    dataset = data.TensorDataset(*data_arrays) #封装数据集张量的类
    return data.DataLoader(dataset, batch_size, shuffle = is_tarin) #返回可迭代对象

batch_size = 10
data_iter = load_array((features,labels),batch_size,is_tarin=True)

data_iter_test = iter(load_array((features,labels),batch_size,is_tarin=True))
#print(next(data_iter_test))

# 模型
net = nn.Sequential(nn.Linear(2,1))
# loss
loss = nn.MSELoss()

# 初始化第0层网络的 weight 和 bias
net[0].weight.data.normal_(0,0.01) 
net[0].bias.data.fill_(0)

# 优化方法
lr = 0.03
trainer = torch.optim.SGD(net.parameters(), lr)
''' net.parameters() 是访问网络参数的生成器 '''
''' trainer 是获得了网络参数的优化器 '''

nums_epoch = 3

for epoch in range(nums_epoch):
    for X,y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad() #清除w,b的梯度
        l.backward()
        trainer.step() #更新网络参数w,b
    l = loss(net(features),labels)
    print(f'epoch {epoch + 1}, loss {l:f}')