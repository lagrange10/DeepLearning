import imp
import torch
from torch import nn
from d2l import torch as d2l
def trans_conv(X,K):
    h,w = K.shape
    Y = torch.zeros(X.shape[0]+h-1,X.shape[1]+w-1)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i:i+h,j:j+w] += K * X[i,j]
    return Y

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(trans_conv(X, K))

#padding增加在输出Y上
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1,1,kernel_size=2,padding=1,bias=False)
tconv.weight.data = K
print(tconv(X))

#stride
tconv = nn.ConvTranspose2d(1,1,kernel_size=2,stride=2,bias=False)
tconv.weight.data = K
print(tconv(X))


"""转置卷积和矩阵乘法"""
X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K)
print(Y)



