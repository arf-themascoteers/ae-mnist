import torch
import torch.nn as nn
from torchvision import datasets, transforms
import autoencoder

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))
    ])

#transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                          batch_size=64,
                                          shuffle=True)

dataiter = iter(data_loader)
images, labels = dataiter.next()
print(torch.min(images), torch.max(images))