import torch
from torch import nn
from torch.nn import functional as F
import os

dir = '../data'

x = torch.arange(4)
torch.save(x, os.path.join(dir,'x-file'))

x2 = torch.load('../data/x-file')
print(x2)

'''张量列表'''
y = torch.zeros(4)
torch.save([x,y],'../data/xy-files')

x2,y2 = torch.load('../data/xy-files')
print(x2,y2)


'''张量字典'''
mydict = {'x':x,'y':y}
torch.save(mydict,'../data/xy-dict')
mydict2 = torch.load('../data/xy-dict')
print(mydict2)

'''加载和保存模型参数'''
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)
    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))

print("--------------------------------------")
net = MLP()
X = torch.randn(size=(2,20))
Y = net(X)
torch.save(net.state_dict(),os.path.join(dir,'mlp.params'))

clone = MLP()
clone.load_state_dict(torch.load(os.path.join(dir,'mlp.params')))
print(clone.eval())

Y_clone = clone(X)
print(Y == Y_clone)