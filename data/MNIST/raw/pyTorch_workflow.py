import torch
from torch import nn
import matplotlib.pyplot as plt

# print(torch.__version__)
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02  # Step size updated to 0.02

x = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * x + bias

print("len(x):", len(x))
print("x[:10]:\n", x[:10])
print("y[:10]:\n", y[:10])

train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]
len(x_train), len(y_train), len(x_test), len(y_test)
