import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root = "../data", train = True, transform = trans, download = True
)
mnist_test = torchvision.datasets.FashionMNIST(
    root = "../data", train = False, transform = trans, download = True
)

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 
    'sandal', 'shirt', 'sneaker','ankle boot']
    return [text_labels[int(i)] for i in labels]


def load_data_fashion_mnist(batch_size, resize = None):
    trans = [transform.ToTensor()]
    if(resize):
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root = "../data", train = True, transform = trans, download = True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root = "../data", train = False, transform = trans, download = True
    )

    return (data.DataLoader(mnist_train, batch_size, shuffle = True, num_workders =
    get_dataloader_workers()), data.DataLoader(mnist_test,batch_size, shuffle = False,
    num_workers = get_dataloader_workers()))