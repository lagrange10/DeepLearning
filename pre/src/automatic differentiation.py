import torch

x = torch.arange(4.0)
x.requires_grad_(True)
print("x=",x)

y = 2 * torch.dot(x,x)
print("y=",y)

y.backward()
print("\n【y=2xTx】\nx.grad=",x.grad)
#pytorch会累积梯度，因此要清楚再进行下一步
x.grad.zero_()
y = x.sum()
y.backward()
print("\n【y=x.sum()】\nx.grad=",x.grad)

#求和计算 一批数据偏导数的和
x.grad.zero_()
y = x * x
print(y)
y.sum().backward()
#y.backward(torch.ones(len(x)))
print("\n【y=x^2.sum()】\nx.grad=",x.grad)

print("\n【分离计算】")
x.grad.zero_()
y = x*x
u = y.detach()
z = u*x
z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()
x.grad == 2 * x

print("\n【控制流梯度计算】")
def f(a):
    # 这个函数最终的结果是根据输入a乘上适当的值，即d = ka k是常数，随着a的分段而改变，分段函数。
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a1 = torch.randn(10, requires_grad=True)
print(a1)
d = f(a1)
print(d)
#d.backward(torch.ones_like(a1)) #
d.sum().backward()
print("【b=f(a)】\na.grad=",a1.grad)
print(a1.grad == d/a1)