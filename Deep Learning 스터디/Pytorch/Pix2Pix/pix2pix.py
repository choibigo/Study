
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import time

parameters = {
    "epoch" : 100,
    "save_epoch" : 5,
    "batch_size" : 2,
    "lr" : 1e-3,
    "train_data_set_1" : "D:\\workspace\\data\\pix2pix\\facades\\train\\a\\",
    "train_data_set_2" : "D:\\workspace\\data\\pix2pix\\facades\\train\\b\\",
    "test_data_set" : "",
    "preprocessing":{
        "resize" : 256,
        "normalize":{
            "maean" : 0.5,
            "stdev" : 0.5
        }
    },
    "using_gpu":True,
}

def RecipeRun(parameter):

    if parameter['using_gpu'] and not torch.cuda.is_available():
        raise Exception("GPU is not avaiable")

    preprocess = preprocessing(parameter['preprocessing'])

    contents_root =  parameter['train_data_set_1']
    style_root =  parameter['train_data_set_2']

    data_uri_list = data_uri(contents_root, style_root)

    dataset = ImageDataset(data_uri_list, preprocess)
    dataloader = DataLoader(dataset, batch_size=parameter['batch_size'], shuffle=True, num_workers=4)

    generator = GeneratorUNet()
    discriminator = Discriminator()

    if parameter['using_gpu']:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # 가중치(weights) 초기화
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # 손실 함수(loss function)
    criterion_GAN = torch.nn.MSELoss() 
    criterion_pixelwise = torch.nn.L1Loss()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

    # 생성자와 판별자를 위한 최적화 함수
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=parameter['lr'])
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=parameter['lr'])

    # 픽셀 단위 Loss인 L1의 차수
    lambda_pixel = 100

    start_time = time.time()

    for epoch in range(1, parameter['epoch']+1):
        for n_epochs, batch in enumerate(dataloader):
            # 모델의 입력(input) 데이터 불러오기
            real_A = batch[1].cuda()
            real_B = batch[0].cuda()

            # 진짜(real) 이미지와 가짜(fake) 이미지에 대한 정답 레이블 생성 (너바와 높이를 16씩 나눈 크기)
            real = torch.cuda.FloatTensor(real_A.size(0), 1, 16, 16).fill_(1.0) # 진짜(real): 1
            fake = torch.cuda.FloatTensor(real_A.size(0), 1, 16, 16).fill_(0.0) # 가짜(fake): 0

            """ 생성자(generator)를 학습합니다. """
            optimizer_G.zero_grad()

            # 이미지 생성
            fake_B = generator(real_A)

            # 생성자(generator)의 손실(loss) 값 계산
            loss_GAN = criterion_GAN(discriminator(fake_B, real_A), real)

            # 픽셀 단위(pixel-wise) L1 손실 값 계산
            loss_pixel = criterion_pixelwise(fake_B, real_B) 

            # 최종적인 손실(loss)
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            # 생성자(generator) 업데이트
            loss_G.backward()
            optimizer_G.step()

            """ 판별자(discriminator)를 학습합니다. """
            optimizer_D.zero_grad()

            # 판별자(discriminator)의 손실(loss) 값 계산
            loss_real = criterion_GAN(discriminator(real_B, real_A), real) # 조건(condition): real_A
            loss_fake = criterion_GAN(discriminator(fake_B.detach(), real_A), fake)
            loss_D = (loss_real + loss_fake) / 2

            # 판별자(discriminator) 업데이트
            loss_D.backward()
            optimizer_D.step()

        print(f"[Epoch {epoch}] [D loss: {loss_D.item():.6f}] [G pixel loss: {loss_pixel.item():.6f}, adv loss: {loss_GAN.item()}] [Elapsed time: {time.time() - start_time:.2f}")
        if epoch % parameter['save_epoch'] == 0:
            img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2) # 높이(height)를 기준으로 이미지를 연결하기
            save_image(img_sample, f"D:\\temp\\inference\\Pix2Pix\\{epoch}.png", nrow=5, normalize=True)

            generator_script = torch.jit.script(generator)
            torch.jit.save(generator_script, f"D:\\temp\\save_model\\Pix2Pix\\{epoch}.pth")
            print("save Model")


#region Generator
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout = 0.0):
        super(UNetDown, self).__init__()
        
        # 너비 & 높이 2배씩 감소
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()

        # 너비와 높이가 2배씩 증가
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1) # 채널 레벨에서 DownSample의 Weight와 concatenation

        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False) # 출력: [64 X 128 X 128]
        self.down2 = UNetDown(64, 128) # 출력: [128 X 64 X 64]
        self.down3 = UNetDown(128, 256) # 출력: [256 X 32 X 32]
        self.down4 = UNetDown(256, 512, dropout=0.5) # 출력: [512 X 16 X 16]
        self.down5 = UNetDown(512, 512, dropout=0.5) # 출력: [512 X 8 X 8]
        self.down6 = UNetDown(512, 512, dropout=0.5) # 출력: [512 X 4 X 4]
        self.down7 = UNetDown(512, 512, dropout=0.5) # 출력: [512 X 2 X 2]
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5) # 출력: [512 X 1 X 1]

        # Skip Connection 사용(출력 채널의 크기 X 2 == 다음 입력 채널의 크기)
        self.up1 = UNetUp(512, 512, dropout=0.5) # 출력: [1024 X 2 X 2]
        self.up2 = UNetUp(1024, 512, dropout=0.5) # 출력: [1024 X 4 X 4]
        self.up3 = UNetUp(1024, 512, dropout=0.5) # 출력: [1024 X 8 X 8]
        self.up4 = UNetUp(1024, 512, dropout=0.5) # 출력: [1024 X 16 X 16]
        self.up5 = UNetUp(1024, 256) # 출력: [512 X 32 X 32]
        self.up6 = UNetUp(512, 128) # 출력: [256 X 64 X 64]
        self.up7 = UNetUp(256, 64) # 출력: [128 X 128 X 128]

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), # 출력: [128 X 256 X 256]
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, kernel_size=4, padding=1), # 출력: [3 X 256 X 256]
            nn.Tanh(),
        )
    def forward(self, x):
        # 인코더부터 디코더까지 순전파하는 U-Net 생성자(Generator)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)
#endregion

# region Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # 두 개의 이미지(실제/변환된 이미지, 조건 이미지)를 입력 받으므로 입력 채널의 크기는 2배
            *self._discriminator_block(in_channels * 2, 64, normalization=False), # 출력: [64 X 128 X 128]
            *self._discriminator_block(64, 128), # 출력: [128 X 64 X 64]
            *self._discriminator_block(128, 256), # 출력: [256 X 32 X 32]
            *self._discriminator_block(256, 512), # 출력: [512 X 16 X 16]
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False) # 출력: [1 X 16 X 16]
        )

    def _discriminator_block(self, in_channels, out_channels, normalization=True):
            # 너비와 높이가 2배씩 감소
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

    # img_A: 실제/변환된 이미지, img_B: 조건(condition)
    def forward(self, img_A, img_B):
        # 이미지 두 개를 채널 레벨에서 연결하여(concatenate) 입력 데이터 생성
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

# endregion

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def data_uri(contents_root, style_root):
    contents_uri_list = glob.glob(contents_root+"\\*.png")
    style_uri_list = glob.glob(style_root+"\\*.png")

    return contents_uri_list, style_uri_list

class ImageDataset(Dataset):
    def __init__(self, uri_list, transform=None):
        self.contents_uri_list = uri_list[0]
        self.style_uri_list = uri_list[1]
        self.transform = transform

    def __getitem__(self, index):

        # Need To Change
        content_image = cv2.imread(self.contents_uri_list[index], 1)
        style_image = cv2.imread(self.style_uri_list[index], 1)
        
        content_image = Image.fromarray(np.uint8(content_image))
        style_image = Image.fromarray(np.uint8(style_image))

        content_image = self.transform(content_image)
        style_image = self.transform(style_image)

        return content_image, style_image

    def __len__(self):
        return len(self.contents_uri_list)

def preprocessing(parameter):
    resize_size = parameter['resize']
    normalize_mean = parameter['normalize']['maean']
    normalize_std = parameter['normalize']['stdev']

    preprocess = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Resize((resize_size, resize_size)),
                transforms.Normalize((normalize_mean, normalize_mean, normalize_mean), (normalize_std, normalize_std, normalize_std))])

    return preprocess



if __name__ == "__main__":
    RecipeRun(parameters)

    print("Success")