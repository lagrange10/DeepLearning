import torch
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

'''Observe the element-wise pairings across the grid, (1, 4),
(1, 5), ..., (3, 6). This is the same thing as the
cartesian product.'''
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
grid_x
'''tensor([[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3]])'''
grid_y
'''tensor([[4, 5, 6],
        [4, 5, 6],
        [4, 5, 6]])'''

'''This correspondence can be seen when these grids are
stacked properly.'''
torch.equal(torch.cat(tuple(torch.dstack([grid_x, grid_y]))),
            torch.cartesian_prod(x, y))
'''True'''

'''`torch.meshgrid` is commonly used to produce a grid for
plotting.'''
import matplotlib.pyplot as plt
xs = torch.linspace(-5, 5, steps=100) #steps分成100份。
ys = torch.linspace(-5, 5, steps=100)
x, y = torch.meshgrid(xs, ys, indexing='xy')
z = torch.sin(torch.sqrt(x * x + y * y))
ax = plt.axes(projection='3d')
ax.plot_surface(x.numpy(), y.numpy(), z.numpy())
plt.show()

