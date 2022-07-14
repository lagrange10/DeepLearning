import matplotlib
import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):
    ''' 人造数据 '''
    X = torch.normal(0,1,(num_examples,len(w))) #随机生成特征的值
    y = torch.matmul(X,w) + b #y是标签值
    y += torch.normal(0,0.01,y.shape) #加噪声
    return X, y.reshape(-1,1) #把y写成列向量

true_W = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = synthetic_data(true_W, true_b, 1000)

print('features:', features[0],'\nlabel:', labels[0])
print(labels[0]-torch.dot(true_W,features[0]))

d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
#d2l.plt.show()

def data_iter(batch_size,features,labels):
    """从数据集中取一批的随机数据"""
    example_nums = len(features)
    indices = list(range(example_nums))
    random.shuffle(indices)
    for i in range(0,example_nums,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,example_nums)]) 
        yield features[batch_indices],labels[batch_indices]

batch_size = 10

for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break

#初始化模型参数
w = torch.normal(0,0.01,size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X,w,b):
    '''线性回归模型'''
    return torch.matmul(X,w) + b

def squared_loss(y_hat, y):
    '''均方损失'''
    return (y_hat - y.reshape(y_hat.shape))**2/2
    # 算出的损失是一个向量，向量元素全部相加是真正的损失函数

def sgd(params, lr, batch_size):  
    """小批量随机梯度下降"""
    # params: 可以是w或者b
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')