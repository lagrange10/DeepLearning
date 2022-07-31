import torch
from torch import nn

def Corr2d(X:torch.Tensor, K:torch.Tensor) -> torch.Tensor:
    """实现二维互相关运算"""
    h,w = K.shape
    Y = torch.zeros(size=(X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w]*K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(Corr2d(X, K))

class Conv2D(nn.Module):
    def __init__(self, kernel_size) -> None:
        super().__init__()
        self.weight = torch.randn(size=kernel_size)
        self.bias = torch.zeros(1)
    def forward(self,X:torch.Tensor) -> torch.Tensor:
        return Corr2d(X,self.weight)+self.bias

X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)

K = torch.tensor([1.0,-1.0]).reshape(1,-1)
Y = Corr2d(X,K)
print(Y)

Z = Corr2d(X.t(),K.t())
print(Z)

conv2d = nn.Conv2d(1,1,(1,2),bias=False)

X = X.reshape((1, 1, 6, 8)) #batch数，通道数，坐标
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2

for i in range(20):
    Y_hat = conv2d(X)
    loss = (Y_hat - Y)**2
    conv2d.zero_grad()
    loss.sum().backward()
    conv2d.weight.data[:] -= lr*conv2d.weight.grad
    if (i+1)%2 == 0:
        print(f"第{i+1}轮 loss = {loss.sum():.3f}")

print(K)
print(Y_hat.reshape(6,7))
