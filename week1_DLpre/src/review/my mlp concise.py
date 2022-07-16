from matplotlib.pyplot import show
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(num_inputs,num_hiddens),
                    nn.ReLU(),
                    nn.Linear(num_hiddens,num_outputs))

def init_weights(m:nn.Linear):
    if nn.Linear == type(m):
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.predict_ch3(net, test_iter)
show()