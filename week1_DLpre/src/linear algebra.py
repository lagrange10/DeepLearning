from matplotlib.pyplot import axis
import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A)

#指定张量沿哪一个轴来通过求和降低维度
A_sum_axis0 = A.sum(axis=0)
print(A_sum_axis0, A_sum_axis0.shape)
print(A.sum(axis=[0,1]))

print("求和但保持维度不变(形式上)，可以用于广播机制：")
sum_A = A.sum(axis = 1)
print(sum_A, sum_A.size())

sum_A = A.sum(axis = 1,keepdim=True)
print(sum_A, sum_A.size())

print(A / sum_A) #广播 实现按行归一化

print("累加求和：")
print(A.cumsum(axis=0))

#张量乘法
a = torch.arange(4,dtype=torch.float32)
x = torch.ones(4,dtype=torch.float32)
print(a.dot(x))
#mv: matrix vector mutiplication
print(A.mv(x))
B = torch.ones(4,3)
print(torch.mm(A,B))

print("向量的L1范数", torch.abs(torch.ones(9)).sum())
print("向量的L2范数", torch.ones(9).norm())
print("矩阵范数", torch.ones(4,9).norm())