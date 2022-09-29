import os
import json
import random
import logging
import time
import shutil
import uuid
import cv2
import numpy as np
import mpp
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR

# Paramter
paramters = ''' {
    "hyperparameter":{
        "epoch" : 100,
        "batch_size" : 4,
        "lr" : 1e-2,
        "using_gpu" : true,
        "using_amp" : true,
        "input_size" : 768,
        "momentum" : 0.9,
        "weight_decay": 1e-4,
        "base_size" : 1024,
        "crop_size" : 768,
        "milestones": [30, 60, 90],
        "gamma": 0.2
    }
}
'''

logger = logging.getLogger('workflow')
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

class SegmentationDataset(data.Dataset):
    def __init__(self, image_path_list, mask_path_list, valid_classes):
        super(SegmentationDataset, self).__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_path_list = image_path_list
        self.mask_path_list = mask_path_list
        assert (len(self.image_path_list) == len(self.mask_path_list))

        self.valid_classes = valid_classes
        self._key = np.full(max(self.valid_classes)+2, -1)
        for i, valid_class in enumerate(self.valid_classes):
            self._key[valid_class+1] = i

        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def __getitem__(self, index):
        img = cv2.imread(self.image_path_list[index], 1)
        mask = cv2.imread(self.mask_path_list[index], 0)
        img = self.transform(img)
        mask = self.__mask_transform(mask)
        return img, mask

    def __mask_transform(self, mask):
        mask = np.array(mask).astype('int32')
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)

        target = self._key[index].reshape(mask.shape)
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.image_path_list)


def sync_preprocess(img, mask, base_size, crop_size):
    crop_size = crop_size
    # random scale (short edge)
    short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
    h = img.shape[0]
    w = img.shape[1]

    if h > w:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    else:
        oh = short_size
        ow = int(1.0 * w * oh / h)

    img = cv2.resize(img, (ow, oh))
    mask = cv2.resize(mask, (ow, oh))

    # pad crop
    if short_size < crop_size:
        padh = crop_size - oh if oh < crop_size else 0
        padw = crop_size - ow if ow < crop_size else 0
        img = cv2.copyMakeBorder(img, 0, padh, 0, padw, cv2.BORDER_CONSTANT, 0)
        mask = cv2.copyMakeBorder(mask, 0, padh, 0, padw, cv2.BORDER_CONSTANT, 0)

    # random crop crop_size
    h = img.shape[0]
    w = img.shape[1]
    x1 = random.randint(0, w - crop_size)
    y1 = random.randint(0, h - crop_size)
    img = img[y1:y1+crop_size, x1:x1+crop_size]
    mask = mask[y1:y1+crop_size, x1:x1+crop_size]

    return img, mask


def download_image(download_path, base_size, crop_size):

    # Need To Change
    image_folder = "D:\\workspace\\data\\ADI_BALL_768\\image\\"
    mask_folder = "D:\\workspace\\data\\ADI_BALL_768\\mask\\"

    image_path_list = list()
    mask_path_list = list()

    for i, file in enumerate(os.listdir(image_folder)):
        origin_image_path = os.path.join(image_folder, file)
        origin_mask_path = os.path.join(mask_folder, file)

        origin_image = cv2.imread(origin_image_path, 1)
        origin_mask = cv2.imread(origin_mask_path, 0)

        image, mask = sync_preprocess(origin_image, origin_mask, base_size, crop_size)

        image_path = os.path.join(download_path, "image", f"{i}.png")
        mask_path = os.path.join(download_path, "mask",f"{i}.png")

        image_path_list.append(image_path)
        mask_path_list.append(mask_path)

        mpp.intel64.save(image, image_path)
        mpp.intel64.save(mask, mask_path)

    return image_path_list, mask_path_list

class FastSCNN(nn.Module):
    def __init__(self, num_classes, aux=False, **kwargs):
        super(FastSCNN, self).__init__()
        self.aux = aux
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


def Trainer(**kwargs):

    if kwargs['hyperparameter']['using_gpu'] and not torch.cuda.is_available():
        raise Exception("GPU is not avaiable")

    hyperparameter = kwargs['hyperparameter']
    device = 'cuda' if kwargs['hyperparameter']['using_gpu'] else 'cpu'

    logger.info("Temp Folder Create")
    local_download_path = os.path.join(f"/temp/{uuid.uuid4()}")
    image_path_list, mask_path_list = download_image(local_download_path,
                                                     hyperparameter['base_size'],
                                                     hyperparameter['crop_size'])

    # Need To Change
    valid_classes = [0, 255]
    train_dataset = SegmentationDataset(image_path_list=image_path_list,
                                        mask_path_list=mask_path_list,
                                        valid_classes=valid_classes)

    
    train_loader = data.DataLoader(dataset=train_dataset,
                                        batch_size=hyperparameter['batch_size'],
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=0)

    # create network
    model = FastSCNN(num_classes=len(valid_classes))  
    model.to(device)

    # create criterion
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                        lr=hyperparameter['lr'],
                                        momentum=hyperparameter['momentum'] ,
                                        weight_decay=hyperparameter['weight_decay'])

    # lr scheduling
    scheduler = MultiStepLR(optimizer, milestones=hyperparameter['milestones'], gamma=hyperparameter['gamma'])

    print('Training Start')
    total_iteration = len(train_loader)
    for epoch in range(1, hyperparameter['epoch']+1):
        model.train()
        scheduler.step(epoch)
        
        iteration_start_time = time.time()
        for n_count, (images, targets) in enumerate(train_loader, 1):
            images = images.to(device)
            targets = targets.to(device)

            with torch.cuda.amp.autocast(enabled=hyperparameter['using_amp']):
                outputs = model(images)
                loss = criterion(outputs[0], targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if n_count % (total_iteration // 10) == 0 or n_count == total_iteration:
                iteration_elapsed_time = time.time() - iteration_start_time
                logger.info(f"Epoch:[{epoch:4d}/{hyperparameter['epoch']:4d}], Iter: [{n_count:4d}/{total_iteration:4d}] || loss: {loss.item():4.4f} || Time: {iteration_elapsed_time:4.2f}s")
                iteration_start_time = time.time()

        model_script = torch.jit.trace(model, images)
        torch.jit.save(model_script, f"D:\\Model_Inference\\save_model\\Fast-Scnn\\{epoch}.pth")

if __name__ == '__main__':
    print('Starting')
    kwargs = json.loads(paramters)
    trainer = Trainer(**kwargs)
