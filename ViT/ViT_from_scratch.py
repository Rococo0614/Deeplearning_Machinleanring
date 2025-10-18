import argparse #这个是是是做什么的?
import math
import os
import time
from typing import Optional #这个又是做什么的?

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

if not hasattr(transforms, "Identity"):
    class Identity:
        def __call__(self, x):
            return x
    transforms.Identity = Identity


def exists(x):
    return x is not None

#这里应该是模型初始化,什么模型的初始化？还是先处理图像?
class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0#图像大写必须能被patch_size，为什么？
        self.num_patches = (img_size // patch_size) ** 2#这里的两个连续符号是矩阵运算？
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride = patch_size)

    def forward(self,x): #(x = B,C,H,W)
        x = self.proj(x)#x的投影？让输入的3通道彩色图像具有步长，核大小的性质？x = (B,embed_dim.H/ps,W/ps)
        x = x.flatten(2).transpose(1,2) #展平x = (B,num_patches,embed_dim)
        return x
    

class MLP(nn.Module): #搭建了多层感知机，并将其中的某些层关闭，只保留输入层)
    def __init__(self,in_features, hidden_features = None, out_features = None,dropout = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features #刚才不要隐藏层和输出层，现在全部初始化为输入形状？
        self.fc1 = nn.Linear(in_features,hidden_features) #线性展开?
        self.act = nn.GELU() #类似于relu的激活函数？
        self.fc2 = nn.Linear(hidden_features,out_features)#这里感觉没什么好说的，线性映射
        self.drop = nn.Dropout(dropout) #训练的时候可以选择遗忘部分神经元，来达到提高训练效果的目的

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
class Attention(nn.Module): #多头自注意力
    def __init__(self,dim,num_heads = 8, qkv_bias = False, attn_drop = 0.8, proj_drop = 0.0):
        super().__init__()
        assert dim % num_heads == 0 #维度能被num_heads 整除
        self.num_heads = num_heads 
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 #为什么要做这个变换?

        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias) #通道进，通道x3出？是把黑白图像变彩色么？
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)#这里输入不是dim*3么？
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,x):
        B,N,C = x.shape
        qkv = self.qkv(x) #从B = batchsize? N = ? C = channels?
        qkv = qkv.reshape(B,N,3,self.num_heads,C // self.num_heads)#???
        qkv = qkv.permute(2,0,3,1,4)#这个函数是作了什么？
        q,k,v = qkv[0],qkv[1],qkv[2]

        attn = (q @ k.transpose(-2,-1)) *self.scale
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module): #编码器
    def __init__(self,dim,num_heads,mlp_ratio = 4.0,
                 qkv_bias = True, drop = 0.0,attn_drop = 0.0, drop_path = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim,eps = 1e-6)
        self.attn = Attention(dim, num_heads=num_heads,qkv_bias=qkv_bias,
                              attn_drop=attn_drop,proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path >0.0 else nn.Identity()#???
        self.norm2 = nn.LayerNorm(dim,eps = 1e-6)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features= dim,hidden_features=mlp_hidden_dim,dropout = drop)

    def forward(self,x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class DropPath(nn.Module):
    def __init__(self,drop_prob = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self,x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype = x.dtype, device = x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
    


class VisionTransformer(nn.Module):
    def __init__(self, img_size = 224, patch_size = 16, in_chans = 3, num_classes = 1000,
                 embed_dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4.0,
                 qkv_bias = True, drop_rate = 0.0, attn_drop_rate = 0.0, drop_path_rate = 0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size = img_size, patch_size = patch_size,
                                      in_chans = in_chans, embed_dim = embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches+1,embed_dim))
        self.pos_drop = nn.Dropout(p = drop_rate)

        dpr = [x.item() for x in torch.linspace(0,drop_path_rate,depth)]

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads,mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim,eps = 1e-6)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std = 0.02)
        nn.init.trunc_normal_(self.cls_token,std = 0.02)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.trunc_normal_(m.weight, std = 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
    
    def forward(self,x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B,-1,-1)
        x = torch.cat((cls_tokens,x),dim =1 )
        x  = x + self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:,0]
        x = self.head(cls)
        return x
    
        
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _,pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0,keepdim=True)
        res.append((correct_k / batch_size).item())
    return res

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scheduler = None, log_every = 50):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0
    for step, (images,targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        output = model(images)
        loss = criterion(output,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step

        bs = images.size(0)
        running_loss += loss.item() * bs
        acc1 = accuracy(output.detach(),targets,topk=(1,))[0]
        running_acc += acc1 * bs
        total += bs

        if step % log_every == 0 and step > 0:
            print(f"Epoch {epoch} Step {step}/{len(dataloader)} loss {running_loss/total:.4f} acc {running_acc/total:.4f}")

    return running_loss / total, running_acc / total


def evaluate(model, dataloader, criterion,device):
    model.eval()
    total = 0
    loss_sum = 0.0
    acc_sum = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            loss = criterion(output, targets)
            bs = images.size(0)
            loss_sum += loss.item() * bs
            acc_sum += accuracy(output,targets,topk=(1,))[0] * bs
            total += bs
    
    return loss_sum / total, acc_sum / total



def get_dataloaders(dataset_name:str, image_size:int, batch_size:int, num_workers=4):
    if dataset_name == 'cifar10':
        mean = (0.4914,0.4822,0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(image_size) if image_size != 32 else transforms.Identity(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(image_size) if image_size != 32 else transforms.Identity(),
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
        ])

        train_ds = datasets.CIFAR10(root='./data',train = True, transform = train_transform, download= True)
        test_ds = datasets.CIFAR10(root='./data',train = False, transform = test_transform,download = True)
        num_classes = 10


        train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle = True, num_workers = num_workers,pin_memory = True)
        test_loader = DataLoader(test_ds, batch_size= batch_size,shuffle = False, num_workers = num_workers, pin_memory = True)
        return train_loader,test_loader,num_classes
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type = str, default = 'cifar10', choices=['cifar10'])
    parser.add_argument('--image-size',type = int, default = 32, help = 'input image size')
    parser.add_argument('--patch-size',type = int,default = 4)
    parser.add_argument('-embed-dim',type=int, default=192)
    parser.add_argument('--depth',type=int, default = 6)
    parser.add_argument('--num-heads', type = int, default = 3)
    parser.add_argument('--mlp-ratio',type=float,default=4.0)
    parser.add_argument('--batch-size',type= int, default=128)
    parser.add_argument('--epochs',type = int, default=20)
    parser.add_argument('--lr',type=float,default=3e-3)
    parser.add_argument('--weight-decay',type = float, default=0.05)
    parser.add_argument('--device',type=str,default='cuda'if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out-dir',type=str,default='./checkpoints')
    parser.add_argument('--log-every',type=int,default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    train_loader, test_loader, num_classes = get_dataloaders(args.dataset, args.image_size, args.batch_size)

    model = VisionTransformer(
        img_size=args.image_size,
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=num_classes,
        embed_dim = args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_rate = 0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr = args.lr,weight_decay = args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = max(1,total_steps))

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss,train_acc = train_one_epoch(model,train_loader,optimizer,criterion,device,epoch,scheduler=None,log_every=args.log_every)
        scheduler.step()
        val_loss, val_acc = evaluate(model, test_loader,criterion,device)
        t1 = time.time()

        print(f"Epoch {epoch} finished in {t1-t0:.1f}s | train_loss {train_loss:.4f} train_acc {train_acc:.4f} | val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {'epoch': epoch, 'model_state': model.state_dict(), 'optim_state': optimizer.state_dict(), 'val_acc': val_acc}
            torch.save(ckpt, os.path.join(args.out_dir, 'best_vit.pth'))
            print(f"Saved best model with val_acc {val_acc:.4f}")
        
if __name__ == '__main__':
    main()