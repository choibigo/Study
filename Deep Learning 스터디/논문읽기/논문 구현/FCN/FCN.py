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

def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf

class BagDataset(Dataset):

    def __init__(self, path, transform=torchvision.transforms.ToTensor()):
       self.transform = transform
       self.origin_path = path+"\\bag\\"
       self.mask_path = path+"\\mask\\"

    def __getitem__(self, idx):
        img_name = os.listdir(self.origin_path)[idx]
        imgA = cv2.imread(self.origin_path+img_name)
        imgA = cv2.resize(imgA, (160, 160))
        imgB = cv2.imread(self.mask_path+img_name, 0)
        imgB = cv2.resize(imgB, (160, 160))
        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.swapaxes(0, 2).swapaxes(1, 2)
        imgB = torch.FloatTensor(imgB)

        if self.transform:
            imgA = self.transform(imgA)    
        item = {'A':imgA, 'B':imgB}
        return item

    def __len__(self):
       return len(os.listdir(self.origin_path))


if __name__ =="__main__":

    save_base_path = "D:\\temp\\save_model\\FCN\\"
    total_epoch = 100
    save_epoch = 5
    batch_size = 4
    lr = 0.01

    vgg_network = VGGNet()
    fcn_network = FCNs(vgg_network, 2)
    fcn_network.cuda()

    criterion = nn.BCELoss()
    # labeling된 이미지와 예측된 이미지 픽셀의 차이를 본다.
    # class가 2개 이므로 Binary Cross Entory를 사용하는듯 하다...?
     
    optimizer = optim.SGD(fcn_network.parameters(), lr = lr, momentum=0.7)

    bag_dataset = BagDataset(path="D:\\workspace\\data\\Segmentation\\bag_data\\")
    dataloader = DataLoader(bag_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    for epoch in range(1, total_epoch + 1):
        epoch_loss = 0.0
        for i, batch in enumerate(dataloader, 1):
            
            # Data
            input = batch['A']
            y = batch['B']

            input = input.cuda()
            y = y.cuda()

            # Predction
            output = fcn_network(input)
            
            # Loss Function
            loss = criterion(output, y) 
            
            # Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch : {epoch}, Iteration : {i}/{int(len(bag_dataset) / batch_size)} Loss : {epoch_loss/i}")

        if epoch % save_epoch == 0:
            torch.save(fcn_network, f"{save_base_path}\\FCN_{epoch}.pth")