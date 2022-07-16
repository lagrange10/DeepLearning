from matplotlib.pyplot import show
import torch
from d2l import torch as d2l
import AccumulatorClass
from Animator import Animator
import matplotlib as plt

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

''' 输入维度看成784, 丢失空间信息——CNN '''
num_inputs = 784
num_outputs = 10

''' 初始化模型参数 '''
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

''' 实现softmax和网络 '''
def softmax(X):
    #定义Softmax
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

''' 实现交叉熵损失 '''
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):  
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: # 如果是个矩阵的话
        y_hat = y_hat.argmax(axis=1)  # 把每行的最大参数拿出来
    cmp = y_hat.type(y.dtype) == y  # 保证数据类型相同(float32?)
    return float(cmp.type(y.dtype).sum()) 

def evaluate_accuracy(net, data_iter):  
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module): #是用模组实现的
        net.eval()
    metric = AccumulatorClass.Accumulator(2)
    with torch.no_grad():  # 因为仅仅要评估训练效果，不要更新网络参数
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):  
    """训练模型一个迭代周期（定义见第3章）
       返回「平均损失」和「平均准确率」
    """
    if isinstance(net, torch.nn.Module): # 如果是用nn模组实现的网络，直接调api就好
        net.train()
    
    metric = AccumulatorClass.Accumulator(3) # [0]:损失函数累积 [1]: 准确率累积 [2]:样本个数
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            '''使用PyTorch内置的优化器和损失函数'''
            updater.zero_grad() #把所有Tensor的梯度清零,防止上一个batch留下的梯度影响
            l.mean().backward() 
            updater.step() #更新参数
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2] # 返回的是「平均损失」 和 「平均准确率」

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater) #train_metrics=(损失，准确率)
        test_acc = evaluate_accuracy(net, test_iter) 
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss   # 训练损失一定不为0，在0和0.5之间
    assert train_acc <= 1 and train_acc > 0.7, train_acc # 训练集准确率一定不为0，在0.7和1之间
    assert test_acc <= 1 and test_acc > 0.7, test_acc # 测试集准确率一定不为0，在0.7和1之间

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

def predict_ch3(net, test_iter, n=6):  
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
show()