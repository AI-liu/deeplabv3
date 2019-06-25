import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np

from dataset import CityScapeDataSet
from deeplabv3 import DeepLabV3
from score import SegmentationMetric

# dataset
batch_size= 2
dataset=CityScapeDataSet()
data_loader = data.DataLoader(dataset, batch_size, shuffle=True , drop_last =True)

# model
model = DeepLabV3()
model.cuda()

# define optimizer
optimizer = optim.Adam(model.parameters())

# define loss function
loss_fn = nn.CrossEntropyLoss()

# Metric
metric = SegmentationMetric(model.num_classes)


for epoch in range(100):
    metric.reset()

    for step , (input_tensor , label_tensor) in enumerate(data_loader):
        input_tensor = Variable(input_tensor).cuda()
        label_tensor = torch.squeeze(label_tensor)
        label_tensor = Variable(label_tensor.long()).cuda()
        

        output = model(input_tensor)
        loss = loss_fn(output,label_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        metric.update(output, label_tensor)
        pixAcc, mIoU = metric.get()
        print("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(step + 1, pixAcc, mIoU))

    if (epoch % 10 == 0):
        torch.save(model, "deeplabv3_"+str(epoch)+ "_mIoU_"+ str(mIoU)[:5] +".pth")




