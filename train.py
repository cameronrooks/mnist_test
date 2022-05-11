import model
from mnist import MNIST
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

training_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

my_nn = model.Net()


#for i, data in enumerate(training_loader):

