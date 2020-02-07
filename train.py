from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import numpy as np
import torch
from net import Net
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim



train, label = loadlocal_mnist(r'data\train-images-idx3-ubyte\train-images.idx3-ubyte', r'data\train-labels-idx1-ubyte\train-labels.idx1-ubyte')
train = torch.from_numpy(train)
labels = torch.from_numpy(label)

# train = train.unsqueeze(0).float / 255.0
# labels = label.unsqueeze(0).long()

# train size: 60000*784, label: 60000

def show(image, label = ''):
    r = np.reshape(image, [28, 28])
    plt.figure()
    plt.imshow(r, cmap='gray')
    plt.title(str(label))
    plt.show()

# for i in range(5):
#     show(train[i], label[i])


net = Net()

# TODO 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# TODO how many epochs, SGD?
for epoch in range(2):
    running_loss = 0.0
    for i in range(len(train)):
        inputs, label = train[i], labels[i]

        inputs = inputs.unsqueeze(0).float() / 255.0
        label = label.long().unsqueeze(0)

        optimizer.zero_grad()
        outputs = net(inputs)


        # print(inputs.size())
        # # print(inputs)
        # print(outputs.size())
        # print(label.size())
        
        # print(outputs)
        # print(label)
        
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('training complete')

# save model
PATH = './MNIST_net.pth'
torch.save(net.state_dict(), PATH)

