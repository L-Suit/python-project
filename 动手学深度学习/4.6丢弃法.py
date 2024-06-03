import torch
from torch import nn
from d2l import torch as d2l


dropout1, dropout2 = 0.2, 0.5

net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

#net = net.to(d2l.try_gpu())


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        # nn.init.normal_(m.weight, std=0.01)


lr = 0.05
batch_size = 512
num_epochs = 10

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none')
loss1 = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
print('train_iter类型：'+str(type(train_iter)))
print(train_iter)
print(trainer)
print(f'net结构:{net}')


d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)