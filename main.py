## Imports
import torch
import torchvision ## Contains some utilities for working with the image data
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
#%matplotlib inline
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn.functional as F
dataset = MNIST(root = 'data/', download = True)
print(len(dataset))
image, label = dataset[10]
plt.imshow(image, cmap = 'gray')
print('Label:', label)
mnist_dataset = MNIST(root = 'data/', train = True, transform = transforms.ToTensor())
print(mnist_dataset)
image_tensor, label = mnist_dataset[0]
print(image_tensor.shape, label)
print(image_tensor[:,10:15,10:15])
print(torch.max(image_tensor), torch.min(image_tensor))
plt.imshow(image_tensor[0,10:15,10:15],cmap = 'gray')
