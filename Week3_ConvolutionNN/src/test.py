import torch
from d2l import torch as d2l
from torch import nn

def corr2d(X:torch.Tensor, K:torch.Tensor) -> torch.Tensor:
    """实现二维互相关运算"""
    h,w = K.shape
    Y = torch.zeros(size=(X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w]*K).sum()
    return Y

def corr2d_multi_in(X, K):
    '''多输入互相关运算'''
    return sum(corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_in_out(X, K):
    '''多输入输出'''
    return torch.stack([corr2d_multi_in(X,k) for k in K], dim=0)

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out(X,K)

conv2d = nn.Conv2d(3,2,(1,1))
conv2d.weight.data = K

Y3 = conv2d(X)
print(Y3,Y1)
print((Y3-Y1).sum())
# assert (Y3-Y1).sum()<1e-6 # 失败！