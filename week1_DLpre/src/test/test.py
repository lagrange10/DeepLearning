import sys
import torch
import random

d1 = torch.tensor([1,2,3,4,5,6,7,8])
indices = list(range(8))
random.shuffle(indices)
batch_indice = torch.tensor(indices)
print(batch_indice)
print(d1[batch_indice])
print(d1[torch.tensor([7,3,2])])

def fibonacci(n): # 生成器函数 - 斐波那契
    a, b, counter = 0, 1, 0
    while True:
        if (counter > n): 
            return
        yield a
        a, b = b, a + b
        counter += 1
f = fibonacci(10) # f 是一个迭代器，由生成器返回生成
 
while True:
    try:
        print (next(f), end=" ")
    except StopIteration:
        sys.exit()

class MyNumbers:
    def giao():
        pass
    def __iter__(self):
        self.a = 1
        return self
 
    def __next__(self):
        x = self.a
        self.a += 1
        return x
 
myclass = MyNumbers()
myiter = iter(myclass)

print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))

def foo(num):
    print("starting...")
    while True:
        num=num+1
        yield num
for n in foo(0):
    print(n)
    break

def giao(*hello):
    return hello[0]+hello[1]

print(giao(1,2))