import torch
from torch import nn
from d2l import torch as d2l

X = torch.ones(2, 3, device=d2l.try_gpu())
print(X)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=d2l.try_gpu())
print(net(X))
print(net[0].weight.data)

