import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import time
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data',train=True,download = False,transform = transform)
test_dataset = datasets.FashionMNIST(root='./data',train = False,download = False, transform = transform)

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers = 4,pin_memory = True)
test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False,num_workers = 4,pin_memory  = True)

classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#定义VGG

class VGG(nn.Module):
    def __init__(self,features,num_classes=10):
        super(VGG,self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512,256),  #之前都是长的变短，这里短的变长是填充？
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(256,num_classes)#直接4096改10？
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

def make_layers(cfg,batch_norm = False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride = 2)]
        else:
            conv2d = nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
            if batch_norm:
                layers += [conv2d,nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
            else:
                layers += [conv2d,nn.ReLU(inplace = True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = [64,64,'M',128,128,'M',256,256,'M',512,512,'M']
vgg = VGG(make_layers(cfg),num_classes=10)
vgg.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg.parameters(),lr = 0.01,momentum=0.9) #权重更新时引入了惯性，能加速收敛，减缓振荡

def train(model,train_loader,criterion,optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_loss = 0
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output,target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _,predicted = output.max(1) #这段代码是什么做法,与predicted = output.argmax(1)有什么区别？第一种写法会出来values,indices两个参数
        total += target.size(0)
        total_loss += loss.item()*data.size(0)
        correct += predicted.eq(target).sum().item()
    return total_loss / len(train_loader.dataset), correct / len(train_loader.dataset)

def test(model,test_loader,criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device), target.to(device)
            output = model(data)
            _,predicted = output.max(1)
            total += target.size(0)
            loss = criterion(output,target)
            total_loss += loss.item()*data.size(0)
            correct += predicted.eq(target).sum().item()
    
    return total_loss / len(test_loader.dataset), correct / len(test_loader.dataset)


model = vgg.to(device)
use_pretrained = True   # 改成 False 就强制重新训练

if use_pretrained and os.path.exists("vgg11_fashionmnist.pth"):
    model.load_state_dict(torch.load("vgg11_fashionmnist.pth"))
    print("加载已有模型 ✅")
            # -------------------------
        # # 随机展示若干测试预测结果
        # # -------------------------
    import random
    model.eval()
    X, y = next(iter(test_loader))
    X, y = X.to(device), y.to(device)
    logits = model(X)
    preds = logits.argmax(dim=1)
    # 取前 8 张
    n_show = 8
    imgs = X[:n_show].cpu()
    preds = preds[:n_show].cpu().numpy()
    labels = y[:n_show].cpu().numpy()
    plt.figure(figsize=(12,3))
    for i in range(n_show):
        plt.subplot(1, n_show, i+1)
        plt.imshow(imgs[i].squeeze(), cmap='gray')
        plt.title(f"P:{classes[preds[i]]}\nT:{classes[labels[i]]}")
        plt.axis('off')
    plt.show()
else:
    print("未找到模型，开始训练 🚀")
    epochs = 10
    train_losses, train_accs, test_losses, test_accs = [],[],[],[]
    start_time = time.time()
    for epoch in range(1, epochs+1):
        t0 = time.time()
        train_loss,train_acc = train(model,train_loader,criterion,optimizer)
        test_loss,test_acc = test(model,test_loader,criterion)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        t1 = time.time()
        print(f"Epoch {epoch:2d}/{epochs} "
              f"train_loss = {train_loss:.4f} train_acc = {train_acc:.4f} "
              f"test_loss = {test_loss:.4f} test_acc = {test_acc:.4f} " 
              f"time = {(t1-t0):.1f}s")
        
    total_time = time.time() - start_time
    print(f"time = {total_time:.1f}s")

        # -------------------------
        # # 可视化训练曲线（在 notebook 中使用 %matplotlib inline）
        # # -------------------------
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="test loss")
    plt.legend()
    plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(train_accs, label="train acc")
    plt.plot(test_accs, label="test acc")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.show()
        # -------------------------
        # # 随机展示若干测试预测结果
        # # -------------------------
    import random
    model.eval()
    X, y = next(iter(test_loader))
    X, y = X.to(device), y.to(device)
    logits = model(X)
    preds = logits.argmax(dim=1)
    # 取前 8 张
    n_show = 8
    imgs = X[:n_show].cpu()
    preds = preds[:n_show].cpu().numpy()
    labels = y[:n_show].cpu().numpy()
    plt.figure(figsize=(12,3))
    for i in range(n_show):
        plt.subplot(1, n_show, i+1)
        plt.imshow(imgs[i].squeeze(), cmap='gray')
        plt.title(f"P:{classes[preds[i]]}\nT:{classes[labels[i]]}")
        plt.axis('off')
        plt.show()
    torch.save(model.state_dict(), "vgg11_fashionmnist.pth")
    print("模型已保存 ✅")





