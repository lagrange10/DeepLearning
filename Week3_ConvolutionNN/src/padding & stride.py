import torch
from torch import conv2d, nn

def Y_conv2d(X:torch.Tensor,conv2d) -> torch.Size:
    X = X.reshape((1,1)+X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

X = torch.rand(8,8)
conv2d = nn.Conv2d(1,1,kernel_size=3,padding=1)
print(Y_conv2d(X,conv2d).shape)

conv2d = nn.Conv2d(1,1,kernel_size=3,padding=1,stride=2)
print(Y_conv2d(X,conv2d).shape)

conv2d = nn.Conv2d(1,1,kernel_size=(3,5),padding=(0,1),stride=(3,4))
print(Y_conv2d(X,conv2d).shape)
