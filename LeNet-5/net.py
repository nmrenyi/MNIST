import torch.nn as nn
from torch.nn.functional import relu

# LeNet-5 https://medium.com/@sh.tsang/paper-brief-review-of-lenet-1-lenet-4-lenet-5-boosted-lenet-4-image-classification-1f5f809dbf17
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 28 -- 24
        self.pool = nn.AvgPool2d((2, 2)) # 24 -- 12
        # self.pool = nn.MaxPool2d((2, 2)) # 24 -- 12

        self.conv2 = nn.Conv2d(6, 16, 5) # 12 -- 8*8
        self.fc1 = nn.Linear(16 * 4 * 4, 120) # 16 * 8 * 8
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        # x = nn.functional.softmax(x, dim = 1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
