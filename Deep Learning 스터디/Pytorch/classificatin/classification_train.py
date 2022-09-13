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
paramters = {
    "epoch" : 50,
    "save_epoch" : 5,
    "batch_size" : 64,
    "lr" : 1e-3,
    "num_classes" : 10,
    "using_gpu" : True,
    "model_save_path":"D:\\temp\\save_model\\GoogleNet\\"
}
def RecipeRun(parameter):
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./Deep Learning 스터디/data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=parameter['batch_size'],
                                            shuffle=True, num_workers=4, drop_last=True)

    model = torchvision.models.googlenet(num_classes = parameter['num_classes'], weights = None)
    if torch.cuda.is_available() and parameter['using_gpu']:
        model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = parameter['lr'])

    input_paramter = next(model.parameters())
    print(input_paramter.shape)

    print("Train Start")
    for epoch in range(1, parameter['epoch']):
        epoch_loss = 0.0
        for i, batch in enumerate(trainloader, 1):
            inputs, labels = batch
            
            if torch.cuda.is_available() and parameter['using_gpu']:
                inputs = inputs.cuda()
                labels = labels.cuda()
            

            print(inputs.shape)
            output, aux1, aux2 = model(inputs)

            optimizer.zero_grad()
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # print(f"Epoch : {epoch} {i}/{int(len(trainset)/parameter['batch_size'])}, Loss : {loss}")

        print(f"Epoch : {epoch}, Loss : {epoch_loss / (len(trainset)/parameter['batch_size'])} ### ")

        if epoch % parameter['save_epoch'] == 0:
            with torch.no_grad():

                model_script = torch.jit.script(model)
                torch.jit.save(model_script, f"{parameter['model_save_path']}\\googlenet_{epoch}.pth")



if __name__ =="__main__":
    RecipeRun(paramters)