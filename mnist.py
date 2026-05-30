import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.ToTensor()

train_data = MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

test_data = MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
)