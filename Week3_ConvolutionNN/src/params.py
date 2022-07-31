import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

print(net[2].state_dict())

print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)


'''批量访问参数'''
print(net[0].named_parameters())
print(next(net[0].named_parameters()))   #其实就是状态字典，但是做成一个生成器
print("----------------------------------\n")
print([(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print([(name, param.shape) for name, param in net.named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

print(type(net.state_dict()['0.bias']))
print(net.state_dict()['0.bias'])
print(type(net.state_dict()['0.bias'].data))
print(net.state_dict()['0.bias'].data)


'''嵌套块'''
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1()) #相比于add 可以自定义名字
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
print(rgnet)


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]

#每块可以用不同的初始化方法
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

'''shared指向的是同一个对象实例'''
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])