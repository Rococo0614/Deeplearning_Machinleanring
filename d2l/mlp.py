import os
import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt


batch_size = 256

transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(), #图幅值转换到0-1
    transforms.Normalize((0.5,),(0.5,)) #知道是从0-1变换为-1-1,但是为什么0.5,?转换成元组,只有在 单元素元组 的时候，必须写上尾随逗号，否则 Python 会把它当作普通的数字

])

train_dataset = datasets.FashionMNIST(root= 'data', train = True, download = False, transform = transform)
test_dataset = datasets.FashionMNIST(root = 'data', train = False, download = False, transform = transform)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True, num_workers = 2, pin_memory = True)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, num_workers = 2, pin_memory = True)

classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#直接用的话，方差比较大，数值容易爆掉，所以才缩小到 0.01 的量级

class mlp(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.classes = num_classes
        self.ReLU = nn.ReLU()

        self.W1 = nn.Parameter(torch.randn(
            input_size,hidden_size,requires_grad = True) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(hidden_size,requires_grad = True))
        self.W2 = nn.Parameter(torch.randn(
            hidden_size,num_classes,requires_grad = True) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(num_classes,requires_grad = True))
    
    def forward(self,X):
            X = X.reshape((-1,self.input_size))
            H = self.ReLU(X@self.W1 + self.b1)
            return(H@self.W2 + self.b2)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = mlp(input_size=784,hidden_size= 256, num_classes= 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

use_pretrained = True

def train(model,train_loader,criterion,optimizer):
    model.train()
    correct = 0
    total_loss = 0
    for data,target in train_loader:
        data,target = data.to(device),target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output,target)

        loss.backward()
        optimizer.step()
        
        predicted = output.argmax(1)

        total_loss += loss.item()*data.size(0)
        correct += (predicted == target).sum().item()
    
    return total_loss / len(train_loader.dataset), correct / len(train_loader.dataset)
#accuracy 只看“对不对”；loss 还看“有多自信”。

def test(model,test_loader,criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output,target)
            predicted = output.argmax(1)

            total_loss += loss.item() * data.size(0)
            total_correct += (predicted == target).sum().item()
        
        return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)



if use_pretrained and os.path.exists("mlp_fashionmnist.pth"):
    model.load_state_dict(torch.load("mlp_fashionmnist.pth"))
    print("加载已有模型")
    import random
    model.eval()
    X,y = next(iter(test_loader))
    X,y = X.to(device), y.to(device)
    logits = model(X)
    preds = logits.argmax(dim=1)

    n_show = 8
    imgs = X[:n_show].cpu()
    preds = preds[:n_show].cpu().numpy()
    labels = y[:n_show].cpu().numpy()
    plt.figure(figsize=(12,3)) #figsize绘图大小，12是1200像素？
    for i in range(n_show):
        plt.subplot(1,n_show,i+1)
        plt.imshow(imgs[i].squeeze(),cmap = 'gray')
        plt.title(f"P:{classes[preds[i]]}\nT:{classes[labels[i]]}")
        plt.axis('off')
    plt.show()
else:
    print("开始训练")
    epochs = 10
    train_losses,train_accs,test_losses,test_accs = [],[],[],[]
    start_time = time.time()
    for epoch in range(1,epochs+1):
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
    torch.save(model.state_dict(), "mlp_fashionmnist.pth")
    print("模型已保存 ✅")

plt.show()
