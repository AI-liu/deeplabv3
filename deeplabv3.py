import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


from aspp import DeepLabHead


import torchvision
import PIL
from torchvision import transforms
import torch.utils.model_zoo as model_zoo
import torchvision.models as models


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def resnet101(pretrained=False):
    model = models.resnet101()
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


class DeepLabV3(nn.Module):
    def __init__(self):
        super(DeepLabV3, self).__init__()

        self.num_classes = 21

        self.backbone = nn.Sequential(*list(resnet101(True).children())[:-2])
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
    input_tensor = transforms.ToTensor()(input_image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to('cuda')
    print(input_batch.shape)   # torch.Size([1, 3, 1024, 2048])
    output = model(input_batch)
    print(output.shape)       # torch.Size([1, 21, 1024, 2048])
