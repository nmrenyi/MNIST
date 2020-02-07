from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import numpy as np
import torch
from net import Net
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

net = Net()
net.load_state_dict(torch.load('MNIST_net_Linear_Model.pth'))


# test, label = loadlocal_mnist(r'data\train-images-idx3-ubyte\train-images.idx3-ubyte', r'data\train-labels-idx1-ubyte\train-labels.idx1-ubyte')

test, label = loadlocal_mnist(r'data\t10k-images-idx3-ubyte\t10k-images.idx3-ubyte', r'data\t10k-labels-idx1-ubyte\t10k-labels.idx1-ubyte')
test = torch.from_numpy(test)
labels = torch.from_numpy(label)

def show(image, label = ''):
    r = np.reshape(image, [28, 28])
    plt.figure()
    plt.imshow(r, cmap='gray')
    plt.title(str(label))
    plt.show()


total = 0
correct = 0

for i in range(len(test)):
    inputs = test[i].unsqueeze(0).float() / 255.0
    # inputs = inputs.view([1, 1, 28, 28])

    label = labels[i].long().unsqueeze(0)
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    total += 1
    correct += (predicted == label).item()
    # result = ''
    # if (predicted == label).item():
    #     result = 'Yes!'
    # else:
    #     result = 'No!'
    # print(f'predicted = {predicted.item()}, label = {label.item()}, {result}')
    

print(f'Accuracy of {len(test)} samples = {correct / (0.0 + total)}')