import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

from resnet import *
from aspp import DeepLabHead


import torchvision
import PIL
from torchvision import transforms



class DeepLabV3(nn.Module):
    def __init__(self):
        super(DeepLabV3, self).__init__()

        self.num_classes = 21

        self.backbone = resnet101() 
        self.classifier = DeepLabHead(2048  ,self.num_classes)
        #self.aux_classifier = FCNHead(self.num_classes) 

    def forward(self, x):
        input_shape = x.shape[-2:]

        feature_map = self.backbone(x) 
        classifier_map = self.classifier(feature_map)
        result = F.interpolate(classifier_map, size=input_shape, mode='bilinear', align_corners=False)
        return result

if __name__ == "__main__":
    print("hello")
    model = DeepLabV3()
    model.cuda()
    model.eval()
    print(model)


    input_image = PIL.Image.open('aachen_000000_000019_leftImg8bit.png')
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to('cuda')
    print(input_batch.shape)   # torch.Size([1, 3, 1024, 2048])
    output = model(input_batch)
    print(output.shape)       # torch.Size([1, 21, 1024, 2048])
