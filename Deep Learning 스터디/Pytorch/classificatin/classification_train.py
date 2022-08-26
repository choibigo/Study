import mpp
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import io
import logging
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim


# Parameters
paramters ={
    "epoch" : 10,
    "save_epoch" : 1,
    "batch_size" : 4,
    "lr" : 1e-3,
    "num_classes" : 10,
    "optimizer":{
        "name" : "Adam",
        "momentum" : 0.9,
        "milestones" : [30, 60, 90],
        "gamma" : 0.2
    },
    "loss":{
        "name" : "crossentropyloss",
    },
    "scheduler":{
        "name" : "multiSteplr"
    },
    "preprocessing":{
        "resize" : 299,
        "normalize":{
            "maean" : 0.5,
            "stdev" : 0.5
        }
    },
    "using_gpu":True,
    "authentication":{
        "user_id":"pipeline",
        "password":"mirero2816!",
        "account_service_address":"192.168.70.32:5010",
        "operation_service_address":"localhost:5020"
    },
    "metadata": {
        "artifact":{
            "artifact_group_id":"dncnn_artifact_group",
            "artifact_group_title":"dncnn_artifact_group_title",
            "artifact_volume_id":"DNCNN-Test" 
        },
        "dataset":{
            "query_parameter":{'page_index': 0,
                       'page_size' : 1,
                       'where' : None,
                       'order_by' : None
                       },
            "lock_timeout_sec": 1.0,
        },
        "chunk_size" : 912
    }
}

logger = logging.getLogger()

def RecipeRun(parameter, dataset_id):

    if parameter['using_gpu'] and not torch.cuda.is_available():
        raise Exception("GPU is not avaiable")

    preprocess = preprocessing(parameter['preprocessing'])
    data_uri = data_generator()
    dataset = ClassificationDataset(data_uri, preprocess)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=parameter['batch_size'],
                                            shuffle=True, num_workers=2)
    
    model = torchvision.models.googlenet(num_classes = parameter['num_classes'])
    if parameter['using_gpu']:
        model = model.cuda()

    criterion = Loss(parameter['loss']['name']).function
    optimizer = Optimizer(parameter['optimizer']['name'], model, parameter['lr']).function

    loss_list = list()

    for epoch in range(1, parameter['epoch'] + 1):  #데이터셋 2번 받기
        epoch_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(trainloader):
            inputs, labels = data

            if parameter['using_gpu']:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            outputs,aux1, aux2 = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            logger.info('Epoch : %4d Batch : %4d / loss = %2.4f' % (
            epoch, i, loss.item() / parameter['batch_size']))

            print('Epoch : %4d Batch : %4d / loss = %2.4f' % (
            epoch, i, loss.item() / parameter['batch_size']))

        elapsed_time = time.time() - start_time
        logger.info(f'epoch = {epoch:4d}, loss = {epoch_loss / i:4.4f}, time = {elapsed_time:4.2f}s')

        loss_list.append(epoch_loss / (i * parameter['batch_size']))

        if epoch % parameter['save_epoch'] == 0:
            loss_image_buffer = loss_graph_image(parameter['epoch'], loss_list)
            # save(model, epoch, loss_image_buffer, operation_channel, access_token, metadata)

    print('Finished Training')

def data_generator():

    uri_list = [r"D:\workspace\data\Classification\CIFAR-10-Flat\0000.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0001.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0002.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0003.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0004.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0005.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0006.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0007.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0008.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0009.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0010.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0011.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0012.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0013.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0014.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0000.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0001.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0002.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0003.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0004.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0005.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0006.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0007.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0008.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0009.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0010.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0011.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0012.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0013.jpg",
                r"D:\workspace\data\Classification\CIFAR-10-Flat\0014.jpg",
                ]

    label_list = ['0','0','0','1','1','1','2','2','2','3','3','3','4','4','4','5','5','5','0','0','0','1','1','1','2','2','2','3','3','3','4','4','4','5','5','5']

    data = (uri_list, label_list)

    return data

class ClassificationDataset(Dataset):
    def __init__(self, uri_label_list, transform):
        super(ClassificationDataset, self).__init__()
        self.uri_list = uri_label_list[0]
        self.label_list = uri_label_list[1]
        self.transform = transform
    
    def __getitem__(self, index):
        
        uri = self.uri_list[index]

        # Need To Change Data
        image = mpp.intel64.load(uri, False)

        # Preprocessing
        image = Image.fromarray(np.uint8(image)) # Numpy TO PIL image
        x = self.transform(image)

        label = self.label_list[index]
        y = torch.tensor(int(label))

        return x, y

    def __len__(self):
        return (len(self.uri_list))

def preprocessing(parameter):
    resize_size = parameter['resize']
    normalize_mean = parameter['normalize']['maean']
    normalize_std = parameter['normalize']['stdev']

    preprocess = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Resize((resize_size, resize_size)),
                transforms.Normalize((normalize_mean, normalize_mean, normalize_mean), (normalize_std, normalize_std, normalize_std))])

    return preprocess

class Loss():
    def __init__(self, loss_name):
        if loss_name.lower() == "mseloss":
            self.function = nn.MSELoss()
        
        elif loss_name.lower() == "bceloss":
            self.function = nn.BCELoss()

        elif loss_name.lower() == "crossentropyloss":
            self.function = nn.CrossEntropyLoss()

        else :
            raise Exception("Invalid Loss Option")

class Optimizer():
    def __init__(self, optimizer_name, model, lr):
        if optimizer_name.lower() == "adam":
            self.function = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name.lower() == "sgd":
            self.function = optim.SGD(model.parameters(), lr=lr)
        elif optimizer_name.lower() == "adagrad":
            self.function = optim.Adagrad(model.parameters(), lr=lr)
        else:
            raise Exception("Invalid Optimizer Option")

def loss_graph_image(total_epoch, loss_list):
    plt.plot(range(1, len(loss_list)+1), loss_list, 'b')
    plt.xticks(list(range(1, total_epoch+1)))
    plt.yticks(list(np.arange(0, math.ceil(loss_list[0]), 0.1)))
    plt.ylabel("loss")
    plt.xlabel("Epoch")

    loss_image_buffer = io.BytesIO()
    plt.savefig(loss_image_buffer, format='png')

    loss_image_value = loss_image_buffer.getvalue()
    return io.BytesIO(loss_image_value)

def log(epoch, epoch_loss, n_count, elapsed_time):
    print('epcoh = %4d , loss = %4.4f , time = %4.2f s' % (epoch, epoch_loss / n_count, elapsed_time))

if __name__ =="__main__":

    dataset_id = 1
    RecipeRun(paramters, dataset_id)