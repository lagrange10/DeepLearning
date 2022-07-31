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
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
 