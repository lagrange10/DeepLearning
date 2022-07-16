from matplotlib.pyplot import show
import torch
from d2l import torch as d2l
from Animator import Animator
from torch import nn

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(data = torch.randn(
    size=(num_inputs,num_hiddens), requires_grad=True) * 0.01) #784*256
b1 = nn.Parameter(data = torch.zeros(num_hiddens), requires_grad=True) #256*1
W2 = nn.Parameter(data = torch.randn(
    size=(num_hiddens,num_outputs), requires_grad=True) * 0.01) #10*256
b2 = nn.Parameter(data = torch.zeros(num_outputs), requires_grad=True) #10*1
params = [W1,b1,W2,b2]

def relu(X:torch.Tensor):
    a = torch.zeros_like(X)
    return torch.max(a,X)

def net(X:torch.Tensor):
    X = X.reshape((-1,num_inputs)) #flatten
    H = relu(X@W1+b1)
    return H@W2+b2

loss = nn.CrossEntropyLoss(reduction='none')
num_epochs, lr = 10, 0.1
trainer = torch.optim.SGD(params=params, lr = lr)

d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)

d2l.predict_ch3(net,test_iter)
show()