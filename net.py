import torch.nn as nn
from torch.nn.functional import relu

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)
    
    def forward(self, x):
        x = self.fc(x)
        # x = nn.functional.softmax(x, dim = 0)
        return x
