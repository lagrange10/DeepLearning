from mimetypes import init
import torch
from torch import Tensor, tensor

class A:
    ...

def toTensor(input:int,*,out:Tensor) -> Tensor:
    data = []
    data.append(input)
    out = tensor(data)
    return out

o = tensor([])
o2 = tensor([2,3,4])
i = 1

print(toTensor(i,out=o))
print("toTensor", toTensor(o2,out=o))
print("o = ", o)
123