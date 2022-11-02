import argparse
import torch
import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import os
from torchvision.utils import make_grid
import numpy as np
from torchvision import transforms
from torch.utils import data
from scipy.ndimage import convolve
import h5py
import json
import logging

##$--
parameters = '''{
    "hyperparameter":{
        "epoch": 10,
        "save_model_epoch":1,
        "batch_size": 2,
        "lr": 0.00001,
        "using_gpu":true,
        "image_size": 128,
        "num_blocks":15,
        "conv_dim":128,
        "scale_factor":2
    }
}
'''
##$--

logger = logging.getLogger('workflow')
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')


torch.set_default_tensor_type(torch.DoubleTensor)


def RecipeRun(**kwargs):
    if kwargs['hyperparameter']['using_gpu'] and not torch.cuda.is_available():
        raise Exception("GPU is not avaiable")

    hyperparameter = kwargs['hyperparameter']
    device = 'cuda' if kwargs['hyperparameter']['using_gpu'] else 'cpu'

    # Need To Remove
    data_dir = 'D:\\workspace\\data\\img_align_celeba_256\\img_align_celeba_256.bk\\'
    # data_dir = 'D:\\workspace\\data\\super_resolution_patch\\'
    dataset = SuperResolutionDataset(data_dir, hyperparameter['image_size'], hyperparameter['scale_factor'])
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=hyperparameter['batch_size'],
                                  shuffle=True,
                                  num_workers=0)


    model = SRMD(hyperparameter['num_blocks'], hyperparameter['conv_dim'], hyperparameter['scale_factor']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), hyperparameter['lr'], [0.5, 0.999])

    criterion = nn.MSELoss()
    for epoch in range(1, hyperparameter['epoch']+1):
        for n_count, (x, y) in enumerate(data_loader):
            model.train()

            x = x.to(device)
            y = y.to(device).to(torch.float64)

            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            logger.info(f"[{n_count}/{len(data_loader)}], Loss: {loss.item()}")

            if (n_count) % 100 == 0:
                model.eval()
                reconst = model(x)

                tmp = nn.Upsample(scale_factor=hyperparameter['scale_factor'])(x.data[:,0:3,:])
                pairs = torch.cat((tmp.data[0:2,:], reconst.data[0:2,:], y.data[0:2,:]), dim=3)
                grid = make_grid(pairs, 2)
                from PIL import Image
                tmp = np.squeeze(grid.data.cpu().data.numpy().transpose((1, 2, 0)))
                tmp = (255 * tmp).astype(np.uint8)
                Image.fromarray(tmp).save(f'D:\\Model_Inference\\inference\\SRMD\\{epoch}_{n_count}.png')

            

        logger.info("[{}/{}] loss: {:.4f}".format(epoch+1, hyperparameter['epoch'], loss.item()))
        if epoch % hyperparameter['save_model_epoch'] == 0:
            model_script = torch.jit.trace(model, x)
            torch.jit.save(model_script, f"D:\\Model_Inference\\save_model\\srmd\\{epoch}.pth")


class Kernels(object):
    def __init__(self, kernels, proj_matrix):
        self.kernels = kernels
        self.P = proj_matrix

        self.kernels_proj = np.matmul(self.P,
                                      self.kernels.reshape(self.P.shape[-1],
                                      self.kernels.shape[-1]))

        self.indices = np.array(range(self.kernels.shape[-1]))
        self.randkern = RandomKernel(self.kernels, [self.indices])

    def RandomBlur(self, image):
        kern = next(self.randkern)
        return Image.fromarray(convolve(image, kern, mode='nearest'))

    def ConcatDegraInfo(self, image):
        image = np.asarray(image)   # PIL Image to numpy array
        h, w = list(image.shape[0:2])
        proj_kernl = self.kernels_proj[:, self.randkern.index - 1]  # Caution!!
        n = len(proj_kernl)  # dim. of proj_kernl

        maps = np.ones((h, w, n))
        for i in range(n):
            maps[:, :, i] = proj_kernl[i] * maps[:, :, i]
        image = np.concatenate((image, maps), axis=-1)
        return image


class RandomKernel(object):
    def __init__(self, kernels, indices):
        self.len = kernels.shape[-1]
        self.indices = indices
        np.random.shuffle(self.indices[0])
        self.kernels = kernels[:, :, :, self.indices[0]]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if (self.index == self.len):
            np.random.shuffle(self.indices[0])
            self.kernels = self.kernels[:, :, :, self.indices[0]]
            self.index = 0

        n = self.kernels[:, :, :, self.index]
        self.index += 1
        return n


class SuperResolutionDataset(data.Dataset):
    def __init__(self, root, image_size, scale_factor):
        self.image_paths = []

        #Need To Remove
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            self.image_paths.extend(glob.glob(os.path.join(root, ext)))

        self.image_size = image_size
        self.scale_factor = scale_factor
        K, P = self.__load_kernels(file_path='kernels/', scale_factor=self.scale_factor)
        self.randkern = Kernels(K, P)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        transform = transforms.Resize((self.image_size * self.scale_factor, self.image_size * self.scale_factor))
        hr_image = transform(image)

        transform = transforms.Compose([
                            transforms.Lambda(lambda x: self.randkern.RandomBlur(x)),
                            transforms.Resize((self.image_size, self.image_size)),
                            transforms.Lambda(lambda x: self.__scaling(x)),
                            transforms.Lambda(lambda x: self.randkern.ConcatDegraInfo(x))
                    ])
        lr_image = transform(hr_image)

        transform = transforms.ToTensor()
        lr_image, hr_image = transform(lr_image), transform(hr_image)

        return lr_image.to(torch.float64), hr_image.to(torch.float64)

    def __len__(self):
        return len(self.image_paths)

    def __scaling(self, image):
        return np.array(image) / 255.0

    def __load_kernels(self, file_path='kernels/', scale_factor=2):
        f = h5py.File(os.path.join(file_path, 'SRMDNFx%d.mat' % scale_factor), 'r')

        directKernel = None
        if scale_factor != 4:
            directKernel = f['net/meta/directKernel']
            directKernel = np.array(directKernel).transpose(3, 2, 1, 0)

        AtrpGaussianKernels = f['net/meta/AtrpGaussianKernel']
        AtrpGaussianKernels = np.array(AtrpGaussianKernels).transpose(3, 2, 1, 0)

        P = f['net/meta/P']
        P = np.array(P)
        P = P.T

        if directKernel is not None:
            K = np.concatenate((directKernel, AtrpGaussianKernels), axis=-1)
        else:
            K = AtrpGaussianKernels

        return K, P


class SRMD(nn.Module):
    def __init__(self, num_blocks=11, conv_dim=128, scale_factor=1, num_channels=18):
        super(SRMD, self).__init__()
        self.num_channels = num_channels
        self.conv_dim = conv_dim
        self.sf = scale_factor

        self.nonlinear_mapping = self.make_layers(num_blocks)

        self.conv_last = nn.Sequential(
                            nn.Conv2d(self.conv_dim, 3*self.sf**2, kernel_size=3, padding=1),
                            nn.PixelShuffle(self.sf),
                            nn.Sigmoid()
                         )

    def forward(self, x):  
        x = self.nonlinear_mapping(x)
        x = self.conv_last(x)
        return x

    def make_layers(self, num_blocks):
        layers = []
        in_channels = self.num_channels
        for i in range(num_blocks):
            conv2d = nn.Conv2d(in_channels, self.conv_dim, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(self.conv_dim), nn.ReLU(inplace=True)]
            in_channels = self.conv_dim

        return nn.Sequential(*layers)


if __name__ == '__main__':
    print("Start")
    
    kwargs = json.loads(parameters)
    RecipeRun(**kwargs)

    print("Success")