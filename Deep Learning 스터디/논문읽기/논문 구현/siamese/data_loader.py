import os
import random
from random import Random

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as dset
from albumentations.pytorch import ToTensorV2
import albumentations as A


def get_train_validation_loader(data_dir, batch_size, num_train, augment, way, trials, shuffle, seed, num_workers,
                                pin_memory):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = dset.ImageFolder(train_dir)
    train_dataset = OmniglotTrain(train_dataset, num_train, augment)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=pin_memory)

    val_dataset = dset.ImageFolder(val_dir)
    val_dataset = OmniglotTest(val_dataset, trials, way, seed)
    val_loader = DataLoader(val_dataset, batch_size=way, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


def get_test_loader(data_dir, way, trials, seed, num_workers, pin_memory):
    test_dir = os.path.join(data_dir, 'test')
    test_dataset = dset.ImageFolder(test_dir)
    test_dataset = OmniglotTest(test_dataset, trials=trials, way=way, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=way, shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)

    return test_loader


# adapted from https://github.com/fangpin/siamese-network
class OmniglotTrain(Dataset):
    def __init__(self, dataset, num_train, augment=False):
        self.dataset = dataset
        self.num_train = num_train
        self.augment = augment
        self.mean = 0.8444
        self.std = 0.5329

    def __len__(self):
        return self.num_train

    def __getitem__(self, index):
        # get image from same class
        if index % 2 == 1:
            label = 1.0
            idx = random.randint(0, len(self.dataset.classes) - 1)
            image_list = [x for x in self.dataset.imgs if x[1] == idx]
            image1 = random.choice(image_list)
            image2 = random.choice(image_list)
            while image1[0] == image2[0]:
                image2 = random.choice(image_list)
        # get image from different class
        else:
            label = 0.0
            image1 = random.choice(self.dataset.imgs)
            image2 = random.choice(self.dataset.imgs)
            while image1 == image2:
                image2 = random.choice(self.dataset.imgs)

        # apply transformation
        if self.augment:
            trans = A.Compose([
                A.Rotate((-15, 15), p=1),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])
        else:
            trans = A.Compose([
                # A.Resize(28, 28),
                # A.Normalize(mean=self.mean, std=self.std),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2(),
            ])


        image1 = cv2.imread(image1[0], cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(image2[0], cv2.IMREAD_GRAYSCALE)
        image1 = cv2.bitwise_not(image1)
        image2 = cv2.bitwise_not(image2)
        image1 = trans(image=image1)['image']
        image2 = trans(image=image2)['image']
        label = torch.from_numpy(np.array(label, dtype=np.float32))

        return image1, image2, label


class OmniglotTest:
    def __init__(self, dataset, trials, way, seed=0):
        self.dataset = dataset
        self.trials = trials
        self.way = way
        self.seed = seed
        self.image1 = None
        self.mean = 0.8444
        self.std = 0.5329

    def __len__(self):
        return self.trials * self.way

    def __getitem__(self, index):
        rand = Random(self.seed + index)
        # get image pair from same class
        if index % self.way == 0:
            label = 1.0
            idx = rand.randint(0, len(self.dataset.classes) - 1)
            image_list = [x for x in self.dataset.imgs if x[1] == idx]
            self.image1 = rand.choice(image_list)
            image2 = rand.choice(image_list)
            while self.image1[0] == image2[0]:
                image2 = rand.choice(image_list)

        # get image pair from different class
        else:
            label = 0.0
            image2 = random.choice(self.dataset.imgs)
            while self.image1[1] == image2[1]:
                image2 = random.choice(self.dataset.imgs)

        trans = A.Compose([
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2(),
        ])

        image1 = cv2.imread(self.image1[0], cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(image2[0], cv2.IMREAD_GRAYSCALE)
        image1 = cv2.bitwise_not(image1)
        image2 = cv2.bitwise_not(image2)
        image1 = trans(image=image1)['image']
        image2 = trans(image=image2)['image']

        return image1, image2, label
