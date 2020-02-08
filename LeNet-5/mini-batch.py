import torch
import torch.utils.data as Data
torch.manual_seed(1) 
MINIBATCH_SIZE = 5    # mini batch size
x = torch.linspace(1, 10, 10)  # torch tensor
y = torch.linspace(10, 1, 10)
print(x)
print(y)
# first transform the data to dataset can be processed by torch
torch_dataset = Data.TensorDataset(x, y)
# put the dataset into DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=MINIBATCH_SIZE,
    shuffle=True,
    # num_workers=2           # set multi-work num read data
)

for epoch in range(3):
    # 1 epoch go the whole data
    for step, (batch_x, batch_y) in enumerate(loader):
        # here to train your model
        print('\n\n epoch: ', epoch, '| step: ', step, '| batch x: ', batch_x.numpy(), '| batch_y: ', batch_y.numpy())
