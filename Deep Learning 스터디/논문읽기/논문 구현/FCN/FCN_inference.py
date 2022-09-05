import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import pdb
import numpy as np 
import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import torchvision
import matplotlib.pyplot as plt

# Backbone Model
class VGGNet(VGG):
    def __init__(self, pretrained=True, model="vgg16", requires_grad=True, remove_fc=True, show_params=False):

        model_architect = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        super().__init__(self._make_layers(model_architect)) 
        self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        # if remove_fc:
        #     del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())
    
    def forward(self, x):
        output = {}
        # Upsampling 과정에서 사용하기 위해 forward 되는 과정의 결과를 저장하고 있다.
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output

    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                # MaxPooling 으로 크기가 절반으로 줄어 든다.
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                # padding을 1하고 kernel_size는 3이기 때문에 Convolution 으로 크기가 줄어들지 않는다.
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

# Fully Convolution Network
class FCNs(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1) # 마지막 결과의 채널은 분류하고자 하는 class의 개수이다.

    def forward(self, x):
        output = self.pretrained_net(x) # classifier 가 제거된 pretrained 모델의 output
        x5 = output['x5'] # size=(N, 512, x.H/32, x.W/32)   # 마지막 레이어의 결과
        x4 = output['x4'] # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3'] # size=(N, 512, x.H/8, x.W/8)     # M 
        x2 = output['x2'] # size=(N, 512, x.H/4, x.W/4)
        x1 = output['x1'] # size=(N, 512, x.H/2, x.W/2)     # 첫번째 레이어의 결과
        
        score = self.bn1(self.relu(self.deconv1(x5))) # size = (N, 512, x.H/16, x.W/16)
        score += x4                                   # element-wise add, size(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score))) # size = (N, 256, x.H/8, x.W/8)
        score += x3                                      # element-wise add, size(N, 512, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score))) # size = (N, 128, x.H/4, x.W/4)
        score += x2                                      # element-wise add, size(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score))) # size = (N, 64, x.H/2, x.W/2)
        score += x1                                      # element-wise add, size(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score))) # size = (N, 32, x.H, x.W)
        score = self.classifier(score)                   # size = (N, n_class, x.H, x.W)

        return nn.functional.sigmoid(score)

if __name__ == "__main__":

    input_path = r"D:\workspace\data\Segmentation\bag_data\bag\26.jpg"
    origin_input_image = cv2.imread(input_path, 1)
    origin_input_image = cv2.resize(origin_input_image, (160,160))
    transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()])

    input_image = transform(origin_input_image)
    input_image = input_image.unsqueeze(dim=0)
    input_image = input_image.cuda()

    model_path = r"D:\temp\save_model\FCN\FCN_90.pth"
    model = torch.load(model_path)
    
    predict = model(input_image)
    predict = predict.squeeze(dim = 0)
    predict = predict.detach().cpu()
    predict = predict.numpy()
    
    predict = (predict * 255).astype(np.uint8)

    object_image = predict[0]
    background_image = predict[1]

    cv2.imwrite("D:\\temp\\inference\\FCN\\object.png", object_image)
    cv2.imwrite("D:\\temp\\inference\\FCN\\background.png", background_image)
    cv2.imwrite("D:\\temp\\inference\\FCN\\input_image.png", origin_input_image)
