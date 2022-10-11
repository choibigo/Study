
import os
import json
import time
import logging
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.optim.lr_scheduler import _LRScheduler
from  torchvision import transforms 
from sklearn.metrics import jaccard_score
import network

parameters = '''{
    "hyperparameter":{
        "epoch" : 100,
        "save_model_epoch" : 1,
        "learning_rate" : 0.01,
        "batch_size" : 2,
        "weight_decay" : 1e-4,
        "model" : "deeplabv3plus_mobilenet",
        "separable_conv" : false,       
        "output_stride" : 16,
        "using_amp": false,


        "train_ratio" : 0.9,
        "validation_save_count" : 10,
        "validation_save_random" : false
    }
}
'''


logger = logging.getLogger('workflow')
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def train(**kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hyperparameter = kwargs['hyperparameter']

    train_path_list, valid_path_list = download_image(hyperparameter['train_ratio'])
    label_info = [0,255]
    num_classes = len(label_info)
    train_dataset = SegmentationDataset(data_path_list=train_path_list,
                                        label_info=label_info)
    train_loader = data.DataLoader(dataset=train_dataset,
                                        batch_size=hyperparameter['batch_size'],
                                        shuffle=False,
                                        drop_last=True,
                                        num_workers=0)
    total_iteration = len(train_loader)

    valid_flag = False
    if len(valid_path_list) != 0:
        valid_flag = True
        valid_dataset = SegmentationDataset(data_path_list=valid_path_list,
                                        label_info=label_info)
        valid_loader = data.DataLoader(dataset=valid_dataset,
                                       batch_size=1,
                                       shuffle=hyperparameter['validation_save_random'],
                                       drop_last=False,
                                       num_workers=0)

    model = network.modeling.__dict__[hyperparameter['model']](num_classes=num_classes, output_stride=hyperparameter['output_stride'])
    model.to(device)

    if hyperparameter['separable_conv'] and 'plus' in hyperparameter['model']:
        network.convert_to_separable_conv(model.classifier)
    set_bn_momentum(model.backbone, momentum=0.01)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * hyperparameter['learning_rate']},
        {'params': model.classifier.parameters(), 'lr': hyperparameter['learning_rate']},
    ], lr=hyperparameter['learning_rate'], momentum=0.9, weight_decay=hyperparameter['weight_decay'])
    scheduler = PolyLR(optimizer, hyperparameter['epoch']*total_iteration, power=0.9)

    print("Train Start")
    train_loss_list = list()

    for epoch in range(1, hyperparameter['epoch']):
        epoch_start_time = time.time()
        train_epoch_loss = 0.0
        valid_epoch_loss = 0.0
        valid_epoch_miou = 0.0

        iteration_start_time = time.time()
        for n_count, (images, labels) in enumerate(train_loader, 1):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=hyperparameter['using_amp']):
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

            if n_count % (total_iteration // 10) == 0 or n_count == total_iteration:
                iteration_elapsed_time = time.time() - iteration_start_time
                logger.info(f"Epoch:[{epoch:4d}/{hyperparameter['epoch']:4d}], Iter: [{n_count:4d}/{total_iteration:4d}], Loss: {loss.item():4.4f}, Time: {iteration_elapsed_time:4.2f}s")
                iteration_start_time = time.time()
            scheduler.step()

        if epoch % hyperparameter['save_model_epoch'] == 0 or epoch == hyperparameter['epoch']:
            model.eval()
            model_script = torch.jit.trace(model, images)
            torch.jit.save(model_script, f"D:\\Model_Inference\\save_model\\DeeplabV3\\{epoch}.pth")
            model.train()

        if valid_flag:
            with torch.no_grad():
                model.eval()
                for valid_n_count, (valid_images, valid_labels) in enumerate(valid_loader):
                    valid_images = valid_images.to(device)
                    valid_labels = valid_labels.to(device)

                    with torch.cuda.amp.autocast(enabled=hyperparameter['using_amp']):
                        valid_outputs = model(valid_images)
                        valid_loss = criterion(valid_outputs, valid_labels)

                    valid_epoch_loss += valid_loss

                    if valid_n_count <= hyperparameter['validation_save_count']:
                        valid_miou = iou_score(outputs, labels, len(label_info))
                        valid_epoch_miou+=valid_miou

        train_loss_list.append(train_epoch_loss/total_iteration)

        epoch_valid_loss_mean = 0 if not valid_flag else valid_epoch_loss/valid_n_count
        epoch_valid_miou_mean = 0 if not valid_flag else valid_epoch_miou/hyperparameter['validation_save_count']
        epoch_elapsed_time = time.time() - epoch_start_time
        logger.info(f"Epoch:[{epoch:4d}/{hyperparameter['epoch']:4d}], "
                f"Train Loss: {train_epoch_loss/n_count:4.4f}, "
                f"Valid Loss : {epoch_valid_loss_mean:4.4f}, "
                f"Valid mIou : {epoch_valid_miou_mean:4.4f}, "
                f"Time: {epoch_elapsed_time:4.2f}s")


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]


def download_image(train_ratio):
    image_folder = "D:\\workspace\\data\\ADI_BALL_768_Aug\\image\\"
    mask_folder = "D:\\workspace\\data\\ADI_BALL_768_Aug\\mask\\"

    image_path_list = list()
    mask_path_list = list()

    for i, file in enumerate(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, file)
        mask_path = os.path.join(mask_folder, file)

        image_path_list.append(image_path)
        mask_path_list.append(mask_path)

    train_len = int(len(image_path_list) * train_ratio)

    return (image_path_list[:train_len], mask_path_list[:train_len]), (image_path_list[train_len:], mask_path_list[train_len:])


class SegmentationDataset(data.Dataset):
    def __init__(self, data_path_list, label_info):
        super(SegmentationDataset, self).__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.image_path_list = data_path_list[0]
        self.mask_path_list = data_path_list[1]
        assert (len(self.image_path_list) == len(self.mask_path_list))

        self.label_info = label_info
        self._key= np.full(max(self.label_info)+2, -1)

        for i, label in enumerate(label_info):
            self._key[label+1] = i

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


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def iou_score(pred, target, label_nums):
    pred = torch.argmax(pred, 1)
    pred = pred.cpu().data.numpy().reshape(-1)
    target = target.cpu().data.numpy().reshape(-1)
    score = jaccard_score(y_true=target, y_pred=pred, labels=[x for x in range(0, label_nums)], average='macro')

    return score 

if __name__ == '__main__':
    
    parameters = json.loads(parameters)
    train(**parameters)
