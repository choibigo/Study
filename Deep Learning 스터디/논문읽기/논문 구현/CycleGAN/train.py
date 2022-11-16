import os
import json
import random
import uuid
import logging
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from models.cycle_gan_model import CycleGANModel

parameters = '''{
    "hyperparameter":{
        "epoch":300,
        "save_model_epoch":1,
        "batch_size":2,
        "lr":0.0002,
        "input_channel":1,
        "output_channel":1,
        "using_gpu": true,
        "latent_vector_size":16,
        "lr_policy": "linear",
        "lr_decay": 30,
        "load_size": 256,
        "crop_size": 256,
        "flip": false,
        "preprocess": "resize_and_crop",
        "train_ratio": 0.8
    }
}
'''

logger = logging.getLogger('workflow')
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def TrainRun(**kwargs):

    if kwargs['hyperparameter']['using_gpu'] and not torch.cuda.is_available():
        raise Exception("GPU is not avaiable")

    hyperparameter = kwargs['hyperparameter']
    hyperparameter['epoch'] = int(hyperparameter['epoch'])
    hyperparameter['batch_size'] = int(hyperparameter['batch_size'])
    device = 'cuda' if kwargs['hyperparameter']['using_gpu'] else 'cpu'

    train_path_list, valid_path_list = download_image(hyperparameter['train_ratio'])

    train_dataset = CycleGANDataset(train_path_list,
                            hyperparameter['load_size'],
                            hyperparameter['crop_size'],
                            hyperparameter['input_channel'],
                            hyperparameter['output_channel'],
                            hyperparameter['preprocess'],
                            hyperparameter['flip'])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=hyperparameter['batch_size'],
                                            num_workers=0,
                                            shuffle=True,
                                            drop_last=True)

    valid_flag = False
    if len(valid_path_list[0]) != 0:
        valid_flag = True
        valid_dataset = CycleGANDataset(valid_path_list,
                                          hyperparameter['load_size'],
                                          hyperparameter['crop_size'],
                                          hyperparameter['input_channel'],
                                          hyperparameter['output_channel'],
                                          hyperparameter['preprocess'],
                                          False)
        
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                            batch_size=2,
                                            num_workers=0,
                                            shuffle=False,
                                            drop_last=True)

    model = CycleGANModel(hyperparameter['lr'],
                            hyperparameter['input_channel'],
                            hyperparameter['output_channel'],
                            device)

    model.setup(hyperparameter['lr_policy'],
                hyperparameter['epoch'],
                hyperparameter['lr_decay'])

    total_iteration = len(train_loader)

    for epoch in range(1, hyperparameter['epoch'] + 1):
        for n_count, data in enumerate(train_loader):
            model.set_input(data)
            model.optimize_parameters()

            # if n_count % (total_iteration // 10) == 0 or n_count == total_iteration:
            #     print(f"Epoch: {epoch}, [{n_count: 4d}/{total_iteration: 4d}], {model.loss_G_GAN: 4.4f}, {model.loss_D: 4.4f},{model.loss_G_GAN2: 4.4f},{model.loss_D2: 4.4f},{model.loss_G_L1: 4.4f},{model.loss_z_L1: 4.4f},{model.loss_kl: 4.4f}")

            if n_count%10 ==0:
                print(f"Epoch: {epoch}, Iter: {n_count}, {model.loss_G}")

                if valid_flag:
                    with torch.no_grad():
                        model.netG_A.eval()
                        model.netD_A.eval()

                        for valid_n_count, valid_data in enumerate(valid_loader):
                            fake_b = model.netG_A(valid_data['A'].to(device))
                            temp = model.netD_A(fake_b)
                            loss_valid_G_GAN = model.criterionGAN(temp, True)
                            # print(f"Epoch: {epoch}, [{valid_n_count: 4d}/{len(valid_loader): 4d}], loss_valid_G_GAN: {loss_valid_G_GAN}")

                            fake_b = fake_b.to('cpu')
                            fake_b = fake_b.detach().cpu().data.numpy()
                            save_image = fake_b[0]
                            save_image = save_image.transpose(1,2,0)
                            min_value =  np.min(save_image)
                            max_value =  np.max(save_image)
                            save_image = ((save_image - min_value) / (max_value - min_value)) * 255
                            save_image = save_image.astype(np.uint8)

                            input_data = valid_data['A'][0]
                            input_image = input_data.detach().cpu().data.numpy()
                            input_image  = input_image.transpose(1,2,0)
                            min_value =  np.min(input_image)
                            max_value =  np.max(input_image)
                            input_image = ((input_image - min_value) / (max_value - min_value)) * 255
                            input_image = input_image.astype(np.uint8)

                            save_base_path = "D:\\Model_Inference\\inference\\"
                            cv2.imwrite(f"{save_base_path}\\cycle_gan_temp\\{epoch}_{valid_n_count}_predict.png", cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR))
                            cv2.imwrite(f"{save_base_path}\\cycle_gan_temp\\{epoch}_{valid_n_count}_input.png", cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))

                        model.netG_A.train()
                        model.netD_A.train()

        # if epoch % hyperparameter['save_model_epoch'] == 0:
        #     model_script = torch.jit.trace(model.netG, (data['A'].to(device), z.to(device)))
        #     torch.jit.save(model_script, f"D:\\Model_Inference\\save_model\\bicycle_gan\\{epoch}.pth")

        model.update_learning_rate()          


def download_image(train_ratio):
    root = "D:\\workspace\\data\\image_to_image_temp\\"
    A_paths = list()
    B_paths = list()
    for path in os.listdir(os.path.join(root, "gds")):
        B_paths.append(os.path.join(root, 'sem', path))
        A_paths.append(os.path.join(root, 'gds', path))

    if len(A_paths) != len(B_paths):
        raise Exception("Not Equal Size A,B Data")

    train_ratio = min(1, train_ratio)
    train_len = int(len(A_paths) * train_ratio)

    return (A_paths[:train_len], B_paths[:train_len]) ,(A_paths[train_len:], B_paths[train_len:])


class CycleGANDataset():
    def __init__(self, ab_paths, load_size, crop_size, output_nc, input_nc, preprocess, no_flip):
        self.load_size = load_size
        self.crop_size = crop_size
        self.preprocess = preprocess
        self.no_flip = no_flip

        assert(self.load_size >= self.crop_size)
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.A_paths = ab_paths[0]
        self.B_paths = ab_paths[1]

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        transform_params = self.get_params(A.size)
        A_transform = self.__get_transform(transform_params, grayscale=(self.input_nc == 1))
        B_transform = self.__get_transform(transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)

    def get_params(self, size):
        w, h = size
        new_h = h
        new_w = w
        if self.preprocess == 'resize_and_crop':
            new_h = new_w = self.load_size
        elif self.preprocess == 'scale_width_and_crop':
            new_w = self.load_size
            new_h = self.load_size * h // w

        x = random.randint(0, np.maximum(0, new_w - self.crop_size))
        y = random.randint(0, np.maximum(0, new_h - self.crop_size))

        flip = random.random() > 0.5

        return {'crop_pos': (x, y), 'flip': flip}

    def __get_transform(self, params=None, grayscale=False, convert=True):
        transform_list = []
        if grayscale:
            transform_list.append(transforms.Grayscale(1))
        if 'resize' in self.preprocess:
            osize = [self.load_size, self.load_size]
            transform_list.append(transforms.Resize(osize))
        elif 'scale_width' in self.preprocess:
            transform_list.append(transforms.Lambda(lambda img: self.__scale_width(img, self.load_size)))

        if 'crop' in self.preprocess:
            if params is None:
                transform_list.append(transforms.RandomCrop(self.crop_size))
            else:
                transform_list.append(transforms.Lambda(lambda img: self.__crop(img, params['crop_pos'], self.crop_size)))

        if self.preprocess == 'none':
            transform_list.append(transforms.Lambda(lambda img: self.__make_power_2(img, base=4)))

        if not self.no_flip:
            if params is None:
                transform_list.append(transforms.RandomHorizontalFlip())
            elif params['flip']:
                transform_list.append(transforms.Lambda(lambda img: self.__flip(img, params['flip'])))

        if convert:
            transform_list += [transforms.ToTensor()]
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)   

    def __scale_width(self, img, target_width):
        ow, oh = img.size
        if (ow == target_width):
            return img
        w = target_width
        h = int(target_width * oh / ow)
        return img.resize((w, h))

    def __crop(self, img, pos, size):
        ow, oh = img.size
        x1, y1 = pos
        tw = th = size
        if (ow > tw or oh > th):
            return img.crop((x1, y1, x1 + tw, y1 + th))
        return img

    def __flip(self, img, flip):
        if flip:
            return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return img                  

    
    def __make_power_2(self, img, base):
        ow, oh = img.size
        h = int(round(oh / base) * base)
        w = int(round(ow / base) * base)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h))

if __name__ == '__main__':

    kwargs = json.loads(parameters)

    print("Start")
    TrainRun(**kwargs)
    print("Success")