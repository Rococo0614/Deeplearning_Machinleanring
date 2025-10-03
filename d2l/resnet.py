import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets,transforms
import time
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(), #图幅值转换到0-1
    transforms.Normalize((0.5,),(0.5,)) #知道是从0-1变换为-1-1,但是为什么0.5,?转换成元组,只有在 单元素元组 的时候，必须写上尾随逗号，否则 Python 会把它当作普通的数字

])

train_dataset = datasets.FashionMNIST(root= 'data', train = True, download = False, transform = transform)
test_dataset = datasets.FashionMNIST(root = 'data', train = False, download = False, transform = transform)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 2, pin_memory = True)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, num_workers = 2, pin_memory = True)

classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train(model,loader,optimizer,criterion,device):
    model.train()
    total_loss, correct = 0,0
    for X,y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*X.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return total_loss/len(loader.dataset),correct/len(loader.dataset)

def test(model,loader,criterion,device):
    model.eval()
    total_loss,correct = 0, 0
    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out,y)
            total_loss += loss.item()*X.size(0)
            correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


class BasicBlock(nn.Module):
    expansion = 1 #定义的这个量是为了表示残差块输出通道数和主分支卷积层输出通道数之间的比例中间通道数小，计算快；最终通道数大，表达力强。做权衡
    def __init__(self,in_channels,out_channels,stride=1): #初始化也不一样，这里把通道数目直接列在了函数里面
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size =3,
                               stride = stride,padding = 1, bias = False) #bias偏置
        self.bn1 = nn.BatchNorm2d(out_channels) #还是归一化，可以有效防止梯度爆炸/梯度消失，同时加快收敛速度
        self.conv2 = nn.Conv2d(out_channels, out_channels,kernel_size = 3,
                                stride = 1, padding = 1 ,bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels) #另一层归一化

        self.shortcut = nn.Sequential() #这还是头一次碰到在这里什么都不干
        if stride != 1 or in_channels != out_channels * self.expansion: #为什么会不是1呢，保证形状对齐，跟bottleneck也有关系
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                kernel_size = 1,stride = stride, bias = False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x))) #这个和外面的ReLU有什么区别？没啥区别
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) #残差加法
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes = 10): #所以是不是我中间再学一步VGG会更好？
        super().__init__()
        self.in_channels = 64
        #1x224x224 out: 64x112x112
        self.conv1 = nn.Conv2d(1,64,kernel_size = 7, stride = 2,
                              padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block,64,num_blocks[0],stride = 1) #这是塞了几个块进来
        self.layer2 = self._make_layer(block,128,num_blocks[1], stride = 2) #步长为什么会变化?
        self.layer3 = self._make_layer(block,256,num_blocks[2],stride = 2)
        self.layer4 = self._make_layer(block,512,num_blocks[3],stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) #这里的输入代表什么？
        self.fc = nn.Linear(512*block.expansion,num_classes)

    def _make_layer(self,block,out_channels,num_blocks,stride):
        strides = [stride] + [1] * (num_blocks - 1) #这里的数字在中括号里
        layers = []
        for s in strides:
            layers.append(block(self.in_channels,out_channels,s)) #block函数是哪里来的?
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers) #layers是一个里面包含block的东西
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.layer1(out) #他把线性层简化，搞成这个样
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    
def ResNet18(num_classes = 10):
        return ResNet(BasicBlock,[2,2,2,2],num_classes) #每个layer里block的数量

model = ResNet18().to(device)  
model = ResNet18().to(device)
try:
    model.load_state_dict(torch.load("resnet18_fashionmnist.pth"))
    print("成功加载已训练好的模型 ✅")
except:
    print("未找到已保存的模型，重新训练 🚀")
criterion = nn.CrossEntropyLoss()
num_epochs = 20
optimizer = optim.Adam(model.parameters(), lr = 0.001)  
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size = 5, gamma = 0.1)


train_losses, train_accs, test_losses, test_accs = [],[],[],[]
start_time = time.time()
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    train_loss,train_acc = train(model,train_loader,optimizer,criterion,device)
    test_loss,test_acc = test(model,test_loader,criterion,device)
    scheduler.step()

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    t1 = time.time()
    print(f"Epoch {epoch:2d}/{num_epochs} "
    f"train_loss = {train_loss:.4f} train_acc = {train_acc:.4f} "
    f"test_loss = {test_loss:.4f} test_acc = {test_acc:.4f} " 
    f"time = {(t1-t0):.1f}s")

total_time = time.time() - start_time
print(f"time = {total_time:.1f}s")


    

# -------------------------
# 可视化训练曲线（在 notebook 中使用 %matplotlib inline）
# -------------------------
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
# 随机展示若干测试预测结果
# -------------------------
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

torch.save(model.state_dict(), "resnet18_fashionmnist.pth")
print("模型已保存 ✅")