import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import time

batch_size = 128
lr = 1e-3
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)) #这一步是做到了什么？Normalize(mean,std) 把0 1变为-1 1
])

train_dataset = datasets.FashionMNIST(root = "./data", train = True, transform = transform, download = False)
test_dataset = datasets.FashionMNIST(root = "./data", train = False, transform = transform, download = False)

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True) 
#加入了新变量 num_workers,是为了使用子进程来并行加载数据，linux多进程支持完善，一般为cpu核数/2
#和pin_memory 把加载出来的张量放在固定内存页，使用时候直接把数据用DMA拷贝到GPU,只会在GPU上训练的时候产生作用
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 2, pin_memory = True)

classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

class AlexNetSmall(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__() #lenet里头可没有后面的num_classes,但其实就是分类,无非是方便后期做更改
        self.features = nn.Sequential(

            #in 1x28x28 out 64x14x14
            nn.Conv2d(1, 64, kernel_size = 5, stride = 1,padding = 2), #灰度图输入，64通道输出，padding2所以等于图像还是28x28
            nn.ReLU(inplace = True), #头一会用ReLU,inplace是什么？选择True或者False改变了是否直接对输入张量进行修改操作，这里后面用不到，但是针对Resnet等有分支的网络，原有数据的保留就是必要的
            nn.MaxPool2d(kernel_size = 2, stride = 2), #最大池化操作，2x2,步长为2，n_pot = (int)((n-k+2p)/s + 1)

            #in 64x14x14 out 192x7x7
            nn.Conv2d(64,192,kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            #in 192x7x7 out 384x7x7
            nn.Conv2d(192,384,kernel_size=3,stride = 1, padding = 1),
            nn.ReLU(inplace = True),

            #in 384x7x7 out 256x7x7
            nn.Conv2d(384,256,kernel_size = 3, stride = 1, padding =1),
            nn.ReLU(inplace = True),

            #in 256x7x7 out 256x3x3
            nn.Conv2d(256,256,kernel_size = 3, stride = 1, padding = 1), 
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
#self.layer1 = self._make_layer(block,64,num_blocks[0],stride = 1)以及之后的四步是类似于四层线性组合提取特征么？还是别的什么新方法？
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  #以0.5的概率随机抛弃，输出设置为0
            nn.Linear(256*3*3, 1024), #2304-1024线性组组合提取特征
            nn.ReLU(inplace = True),
            nn.Dropout(0.5), #再丢一次
            nn.Linear(1024,512),
            nn.ReLU(inplace = True),
            nn.Linear(512,num_classes)
        )
    

    def forward(self,X):
        X = self.features(X)  #上面函数就写的这个
        X = torch.flatten(X, 1)
        X = self.classifier(X)
        return X

model = AlexNetSmall(num_classes = 10).to(device)
print(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr) #Adam算子，相较于SGD有新的变化Adam 会维护梯度的 一阶矩（均值）+ 二阶矩（方差），对不同参数自适应调整步长,更平滑

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 6, gamma = 0.5) #决定每一步迈多大，先大步走确定大方向，然后慢慢细调整

def evaluate(model,dataloader,device):
    model.eval() #只是切换模式，在lenet中由于你没有写dropout，也没有写batch_norm，所以训练和测试没啥区别,主要是保持习惯
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad(): #测试不需要记录梯度图
     for X,y in dataloader:
        X,y = X.to(device), y.to(device)
        logits = model(X) #分类结果 10x y.size()
        loss = criterion(logits,y)  #回去重看交叉熵损失
        loss_sum += loss.item() * y.size(0) #每一个测试例都有一个损失
        preds = logits.argmax(dim = 1) #dim = 1是什么？出现俩概率相同的？
        correct += (preds == y).sum().item()
        total += y.size(0) #所有的测试用例加起来？有多少个?

    return loss_sum / total, correct / total

def train(model,dataloader,optimizer,device):
    model.train() #只要你切换，决定了model里的dropout是否执行，省的再来条件语句
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for X, y in dataloader:
        X,y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits,y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim = 1)
        running_correct += (preds == y).sum().item()
        running_total += y.size(0)
    return running_loss / running_total, running_correct / running_total


train_losses, train_accs, test_losses, test_accs = [],[],[],[]
start_time = time.time()
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    train_loss,train_acc = train(model,train_loader,optimizer,device)
    test_loss,test_acc = evaluate(model,test_loader,device)
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