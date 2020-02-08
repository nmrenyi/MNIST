import torch.nn as nn
from torch.nn.functional import relu

# Linear Model, Accuracy = 91.03%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.fc(x)
        x = nn.functional.softmax(x)
        return x

# # One-Hidden-Layer Fully Connected Multilayer NN
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 300)
#         self.fc2 = nn.Linear(300, 10)
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = nn.functional.softmax(x, dim = 1)
#         return x


# # Two-Hidden-Layer Fully Connected Multilayer NN
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 300)
#         self.fc2 = nn.Linear(300, 150)
#         self.fc3 = nn.Linear(150, 10)
#     def forward(self, x):
#         x = relu(self.fc1(x))
#         x = relu(self.fc2(x))
#         x = relu(self.fc3(x))
#         x = nn.functional.softmax(x, dim = 1)
#         return x

