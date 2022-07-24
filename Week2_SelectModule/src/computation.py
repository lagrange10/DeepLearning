import torch
from torch import ne, nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
print(X)
print(X.shape)
print("net",net(X))


class MLP(nn.Module):
    '''自定义一个多层感知机'''
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)

    def forward(self,X) -> torch.Tensor:
        return self.out(F.relu(self.hidden(X)))

net = MLP()
print(net(X))



class MySequential(nn.Module):
    '''实现nn.Sequential'''
    def __init__(self, *args) -> None:
        super().__init__()
        for block in args:
            self._modules[block] = block # 一个顺序字典

    def forward(self, X) -> torch.Tensor:
        """forward会调用__call__"""
        for block in self._modules.values():
            X = block(X)
        return X

net = MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
print("MySequential: ", net(X))

#各种组合使用
class FixedHiddenMLP(nn.Module):
    '''固定参数的隐藏层'''
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.rand(20,20, requires_grad=False)
        self.linear = nn.Linear(20,20)
    def forward(self,X) -> torch.Tensor:
        X = self.linear(X)
        X = F.relu(torch.mm(X,self.weight))
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net = FixedHiddenMLP()
print("FixedHiddenMLP: ", net(X))


class NestMLP(nn.Module):
    """混合搭配各种块"""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),
                                nn.Linear(64,32),nn.ReLU())
        self.linear = nn.Linear(32,16)
    def forward(self,X:torch.Tensor):
        X = self.net(X)
        X = self.linear(X)
        return X

net = nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
print("NestMLP: ", net(X))