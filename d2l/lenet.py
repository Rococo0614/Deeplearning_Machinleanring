import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 准备数据

batch_size = 256 #一次迭代拿出的样本数
transform = transforms.ToTensor() #灰度图，把uint8(0-255)的图像的亮度转换成0-1方便计算

train_dataset = datasets.FashionMNIST(
    root = "./data", train = True, transform = transform, download = True #第一个true/false是该数据集已经分类好了，相当于该进哪个文件夹去东西
)
test_dataset = datasets.FashionMNIST(
    root = "./data", train = False, transform = transform, download = True
)

train_loader = DataLoader(train_dataset,batch_size = batch_size, shuffle = True) #打乱数据顺序，防止看到一堆一类的东西，学到本不该学的规律
test_loader = DataLoader(test_dataset,batch_size = batch_size, shuffle = False)

classes = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
           'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

class LeNet(nn.Module): #nn.Module是模板基类，包括类似前向传播接口，parameters()一键拿到所有参数
    #使用继承并初始化
    def __init__(self): 
        super().__init__()
        #定义conv有以下这么多层:
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,kernel_size = 5, padding = 2), #灰度图，输入通道是1，输出通道是6(代表了学习到六种不同的特征),核是5x5,
            #padding保证输出还是28x28
            nn.Sigmoid(), #激活函数
            nn.AvgPool2d(kernel_size=2,stride = 2), #核是2x2,步长是2 这里不改变通道数量，池化
            nn.Conv2d(6,16,kernel_size = 5), #这里不用padding,逼迫小图提取更高层特征,并且转成16种特征输出
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2,stride =2)#第二层池化,最后得到
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*5*5,120), #16通道数，经过卷积和池化后变成5*5是400什么意思？400-120线性组合后提出特征
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84,10)   #最后84转10分类
        )
    
    def forward(self,x):    #LeNet只是一个层，torch以给好backward
        x = self.conv(x)
        x = self.fc(x)
        return x

def train_epoch(model, dataloader, loss_fn,optimizer,device):
    model.train() #这个train又是做了告诉系统我是在做训练，需要随即丢神经元防止过拟合并做归一化
    total_loss,total_correct = 0,0
    for X, y in dataloader: #for X,y in 应该是循环体,一张图对应一个输出呗
        X,y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred,y) #这不是新定义的么？ 怎么算?:因为正式执行的时候会先走nn.CrossEntropyLoss，所以就学到了交叉熵损失

        optimizer.zero_grad() #清零
        loss.backward() ##沿着loss反向计算参数的梯度
        optimizer.step() #更新参数的方法

        total_loss += loss.item() * y.size(0) #size(0)是什么,取batch_size的个数，由于最后可能是%256,所以最后会有剩余，这里就不能直接用batchsize
        total_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    return total_correct / len(dataloader.dataset), total_correct / len(dataloader.dataset)


 #所以模型是训练好直接拿来用
def test_epoch(model,dataloader,loss_fn,device):
    model.eval() #与之前的train()相对应
    total_loss,total_correct = 0, 0
    #with是做什么用？
    with torch.no_grad(): 
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred,y)
            total_loss += loss.item() * y.size(0)
            total_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        return total_loss / len(dataloader.dataset), total_correct / len(dataloader.dataset)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = LeNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.9)

num_epochs = 20
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model,train_loader,loss_fn,optimizer,device)
    test_loss, test_acc = test_epoch(model,test_loader,loss_fn,device)
    print(f"Epoch {epoch+1}: "
    f"Train loss {train_loss:.4f}, acc {train_acc:.4f}"
    f"Test loss {test_loss:.4f}, acc {test_acc:.4f}")
plt.show()

