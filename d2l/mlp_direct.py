import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


net = nn.Sequential(nn.Flatten(),
nn.Linear(784,256),nn.ReLU(),nn.Linear(256,10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std = 0.01)


# 在 train_ch13 里只用 net.to(device)，不要 DataParallel
device = d2l.try_gpu()  # 或 torch.device('cuda:0')
net = net.to(device)
net.apply(init_weights)

batch_size, lr, num_epochs = 256,0.1,10
loss = nn.CrossEntropyLoss(reduction = 'none')
trainer = torch.optim.SGD(net.parameters(),lr = lr)

train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs)
plt.show()