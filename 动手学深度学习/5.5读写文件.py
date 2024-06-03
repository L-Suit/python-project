import torch
from torch import nn
from torch.nn import functional as F


# 保存向量
x = torch.arange(7)
torch.save(x, '../data/test/x-file')

x2 = torch.load('../data/test/x-file')
print(x2)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
print(Y)

# 保存模型
torch.save(net.state_dict(), '../data/test/mlp.params')

# 加载模型
clone = MLP()
clone.load_state_dict(torch.load('../data/test/mlp.params'))
print(clone.eval())

Y_clone = clone(X)
print(Y == Y_clone)


