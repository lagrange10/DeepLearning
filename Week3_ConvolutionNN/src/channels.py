import torch
from d2l import torch as d2l
from torch import nn

def corr2d_multi_in(X, K):
    '''多输入互相关运算'''
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(corr2d_multi_in(X, K))

def corr2d_multi_in_out(X, K):
    '''多输入输出'''
    return torch.stack([corr2d_multi_in(X,k) for k in K], dim=0)

K = torch.stack([K,K+1,K+2],dim = 0)
print(K.shape)
print(corr2d_multi_in_out(X,K))

def corr2d_multi_in_out_1x1(X, K):
    '''1x1卷积'''
    c_in, h, w = X.shape
    c_out = K.shape[0]
    X = X.reshape(c_in,h*w)
    K = K.reshape(c_out,c_in)
    Y = torch.matmul(K,X).reshape(c_out,h,w)
    return Y

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X,K)
Y2 = corr2d_multi_in_out(X,K)
print((Y1-Y2).sum())

conv2d = nn.Conv2d(3,2,(1,1))
conv2d.weight.data = K


Y3 = conv2d(X)
print(Y3,Y1)
print((Y3-Y1).sum())
# assert (Y3-Y1).sum()<1e-6 # 失败！