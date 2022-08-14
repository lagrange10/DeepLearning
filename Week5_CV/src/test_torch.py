import torch

a=torch.arange(12)
b = a + 1
c = a + 2
ex1 = torch.stack([a,b,c],dim=0)
ex2 = torch.stack([a,b,c],dim=1)
ex3 = ex2.repeat_interleave(5,dim=0)
# print(ex1)
print(ex2, ex2.shape)
print(ex3, ex3.shape)
# a = a.reshape(3,4)
# b=a+1
# c=a+2
# ex3 = torch.stack([a,b,c],dim=0)
# ex4 = torch.stack([a,b,c],dim=1)
# ex5 = torch.stack([a,b,c],dim=2)

# print(ex3, ex3.shape)
# print(ex4, ex4.shape)
# print(ex5, ex5.shape)


