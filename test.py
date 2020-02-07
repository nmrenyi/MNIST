from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import numpy as np
import torch
from net import Net
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

batch_size = 5
nb_classes = 2
in_features = 10

model = nn.Linear(in_features, nb_classes)
criterion = nn.CrossEntropyLoss()

x = torch.randn(batch_size, in_features)
target = torch.empty(batch_size, dtype=torch.long).random_(nb_classes)
print(x)
print(target)
output = model(x)
loss = criterion(output, target)
loss.backward()
