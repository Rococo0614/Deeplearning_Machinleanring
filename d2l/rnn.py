import os
import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

batch_size = 128
transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(root = './data', train = True, download = False, transform = transform) #没找到，transform是干啥的来这？
test_data = datasets.FashionMNIST(root = './data', train = False, download = False, transform = transform)

train_loader = DataLoader(train_data,batch_size=batch_size,shuffle = True, num_workers= 4, pin_memory= True)
test_loader = DataLoader(test_data,batch_size=batch_size,shuffle = False,num_workers=4,pin_memory= True)

classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

class MyRNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        #权重参数的实现。目前只是简单的一层。
        #每次读一行图像，都会结合“当前行的特征”和“上一行的记忆”去更新隐藏状态

        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01) #输入层到隐藏从层的权重
        self.W_hh = nn.Parameter(torch.randn(hidden_size,hidden_size) * 0.01) #上一时刻隐藏状态到当其时刻隐藏状态的权重
        self.b_h = nn.Parameter(torch.zeros(hidden_size))#隐藏层的偏置
        self.W_hq = nn.Parameter(torch.randn(hidden_size,num_classes)*0.01)#隐藏层到输出层的权重
        self.b_q = nn.Parameter(torch.zeros(num_classes))#输出层的偏置


#第1层学习局部依赖（相邻像素/相邻时间步）；第2层学习更长的依赖；第3层学习整体趋势。等
    def forward(self,X):
        X = X.squeeze(1) #为什么要压到1？
        H = torch.zeros(X.shape[0],self.hidden_size,device = X.device)

        for t in range(X.shape[1]):
            X_t = X[:,t,:]
            H = torch.tanh(X_t @ self.W_xh + H @ self.W_hh + self.b_h) #这里使用了tanh激活函数
        
        y = H @ self.W_hq + self.b_q
        return y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyRNN(input_size=28,hidden_size= 128, num_classes= 10).to(device)
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
        for data,target in train_loader:
            data,target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output,target)
            predicted = output.argmax(1)

            total_loss += loss.item() * data.size(0)
            total_correct += (predicted == target).sum().item()
        
        return total_loss / len(test_loader.dataset), total_correct / len(test_loader.dataset)



if use_pretrained and os.path.exists("Rnn_fashionmnist.pth"):
    model.load_state_dict(torch.load("Rnn_fashionmnist.pth"))
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
    torch.save(model.state_dict(), "Rnn_fashionmnist.pth")
    print("模型已保存 ✅")




