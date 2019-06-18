import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np

from dataset import CityScapeDataSet
from deeplabv3 import DeepLabV3

# dataset
batch_size= 2
dataset=CityScapeDataSet()
data_loader = data.DataLoader(dataset, batch_size, shuffle=True)

# model
model = DeepLabV3()
model.cuda()

# define optimizer
optimizer = optim.Adam(model.parameters())

# define loss function
loss_fn = nn.CrossEntropyLoss()


for epoch in range(200):
    for batch_idx , (input_tensor , label_tensor) in enumerate(data_loader):
        optimizer.zero_grad()

        input_tensor = Variable(input_tensor).cuda()
        label_tensor = torch.squeeze(label_tensor)
        label_tensor = Variable(label_tensor.long()).cuda()

        output = model(input_tensor)
        loss = loss_fn(output,label_tensor)
        print(loss)
        loss.backward()
        optimizer.step()

        if (epoch % 50 == 0):
            torch.save(model, "deeplabv3_"+str(epoch)+".pth")




