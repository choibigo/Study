import argparse
import os
import math
import random
import numpy as np
import cv2

from PIL.Image import RASTERIZE
from PIL import Image
import PIL.Image as pil_image

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch import nn
from torch.cuda import amp
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.utils import spectral_norm
from scipy import special
from scipy import ndimage
from scipy.stats import multivariate_normal
from scipy.linalg import orth

class degradation_params:
    # first degradation
    fd_blur_kernel_size = 21
    fd_kernel_list = [
            "iso",
            "aniso",
            "generalized_iso",
            "generalized_aniso",
            "plateau_iso",
            "plateau_aniso",
        ]
    fd_kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    fd_blur_sigma = [0.2, 3]
    fd_betag_range = [0.5, 3]
    fd_betap_range = [1, 2]
    fd_sinc_prob = 0.1
    fd_updown_type = ["up", "down", "keep"]
    fd_mode_list = ["area", "bilinear", "bicubic"]
    fd_resize_prob = [0.2, 0.7, 0.1]
    fd_resize_range = [0.15, 1.5]

    fd_is_gen_kernel = True
    fd_is_sinc = True
    fd_is_randomresizing = True
    fd_is_add_gaussian_noise = True
    fd_is_add_poisson_noise = True
    fd_is_add_jpeg_noise = True

    # second degradation
    sd_blur_kernel_size = 21
    sd_kernel_list = [
            "iso",
            "aniso",
            "generalized_iso",
            "generalized_aniso",
            "plateau_iso",
            "plateau_aniso",
        ]
    sd_kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    sd_blur_sigma = [0.2, 1.5]
    sd_betag_range = [0.5, 4]
    sd_betap_range = [1, 2]
    sd_sinc_prob = 0.1
    sd_resize_prob = [0.3, 0.4, 0.3]
    sd_resize_range = [0.3, 1.2]

    sd_is_gen_kernel = True
    sd_is_sinc = True
    sd_is_randomresizing = True
    sd_is_add_gaussian_noise = True
    sd_is_add_poisson_noise = True
    sd_is_add_jpeg_noise = True

    final_sinc_prob = 0.8


# models.loss.py
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))

        self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class VGGLoss(torch.nn.Module):
    def __init__(self, feature_layer: int = 35) -> None:
        super(VGGLoss, self).__init__()
        model = torchvision.models.vgg19(pretrained=False)

        current_path = os.path.dirname(os.path.realpath(__file__))
        pretrained_model_path = os.path.join(current_path, 'vgg19-dcbb9e9d.pth')
        model.load_state_dict(torch.load(pretrained_model_path))

        self.features = torch.nn.Sequential(
            *list(model.features.children())[:feature_layer]
        ).eval()
        # Freeze parameters. Don't train.
        for name, param in self.features.named_parameters():
            param.requires_grad = False

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        vgg_loss = torch.nn.functional.l1_loss(
            self.features(source), self.features(target)
        )

        return vgg_loss

def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)

    p2d_h = (0, 0, 1, 0)
    p2d_w = (1, 0, 0, 0)

    if hh % 2 != 0:
        x = F.pad(x, p2d_h, "reflect")
    if hw % 2 != 0:
        x = F.pad(x, p2d_w, "reflect")
    h = x.shape[2] // scale
    w = x.shape[3] // scale

    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1
        )

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x



class Generator(nn.Module):
    def __init__(
        self,
        scale=4,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
    ):
        super(Generator, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    @amp.autocast()
    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class Discriminator(nn.Module):

    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):
        super(Discriminator, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm

        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # extra
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))

        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)
    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


# utils.py
def check_image_file(filename: str):
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP"])

def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# degradation.py
class Degradation:
    def __init__(self, dparam, sf=4):
        self.sf = sf
        self.blur_kernel_size = dparam.fd_blur_kernel_size
        self.kernel_list = dparam.fd_kernel_list
        self.kernel_prob = dparam.fd_kernel_prob
        self.blur_sigma = dparam.fd_blur_sigma
        self.betag_range = dparam.fd_betag_range
        self.betap_range = dparam.fd_betap_range
        self.sinc_prob = dparam.fd_sinc_prob
        self.updown_type = dparam.fd_updown_type
        self.mode_list = dparam.fd_mode_list
        self.resize_prob = dparam.fd_resize_prob
        self.resize_range = dparam.fd_resize_range

        self.is_gen_kernel = dparam.fd_is_gen_kernel
        self.is_sinc = dparam.fd_is_sinc
        self.is_randomresizing = dparam.fd_is_randomresizing
        self.is_add_gaussian_noise = dparam.fd_is_add_gaussian_noise
        self.is_add_poisson_noise = dparam.fd_is_add_poisson_noise
        self.is_add_jpeg_noise = dparam.fd_is_add_jpeg_noise

        # blur settings for the second degradation
        self.blur_kernel_size2 = dparam.sd_blur_kernel_size
        self.kernel_list2 = dparam.sd_kernel_list
        self.kernel_prob2 = dparam.sd_kernel_prob
        self.blur_sigma2 = dparam.sd_blur_sigma
        self.betag_range2 = dparam.sd_betag_range
        self.betap_range2 = dparam.sd_betap_range
        self.sinc_prob2 = dparam.sd_sinc_prob
        self.resize_prob2 = dparam.sd_resize_prob
        self.resize_range2 = dparam.sd_resize_range

        self.is_gen_kernel2 = dparam.sd_is_gen_kernel
        self.is_sinc2 = dparam.sd_is_sinc
        self.is_randomresizing2 = dparam.sd_is_randomresizing
        self.is_add_gaussian_noise2 = dparam.sd_is_add_gaussian_noise
        self.is_add_poisson_noise2 = dparam.sd_is_add_poisson_noise
        self.is_add_jpeg_noise2 = dparam.sd_is_add_jpeg_noise

        self.final_sinc_prob = dparam.final_sinc_prob

        self.kernel_range = [
            2 * v + 1 for v in range(3, 11)
        ]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(
            21, 21
        ).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def uint2single(self, img):
        return np.float32(img / 255.0)

    def single2uint(self, img):
        return np.uint8((img.clip(0, 1) * 255.0).round())

    def degradation_pipeline(self, image):
        image = self.uint2single(np.array(image))
        hq = image.copy()

        if self.is_gen_kernel == True:
            image = self.generate_kernel1(image)
        if self.is_sinc == True:
            image = self.generate_sinc(image)
        if self.is_randomresizing == True:
            image = self.random_resizing1(image)
        if self.is_add_gaussian_noise == True:
            image = self.add_Gaussian_noise(image)
        if self.is_add_poisson_noise == True:
            image = self.add_Poisson_noise(image)
        if self.is_add_jpeg_noise == True:
            image = self.add_JPEG_noise(image)

        if self.is_gen_kernel2 == True:
            image = self.generate_kernel2(image)
        if self.is_randomresizing2 == True:
            image = self.random_resizing2(image)
        if self.is_add_poisson_noise2 == True:
            image = self.add_Poisson_noise(image)
        if self.is_add_gaussian_noise2 == True:
            image = self.add_Gaussian_noise(image)

        if np.random.uniform() < 0.5:
            if self.is_sinc2 == True:
                image = self.generate_sinc(image)
            if self.is_add_jpeg_noise2 == True:
                image = self.add_JPEG_noise(image)
        else:
            if self.is_add_jpeg_noise2 == True:
                image = self.add_JPEG_noise(image)
            if self.is_sinc2 == True:
                image = self.generate_sinc(image)

        # resize to desired size
        image = cv2.resize(
            image,
            (int(1 / self.sf * hq.shape[1]), int(1 / self.sf * hq.shape[0])),
            interpolation=random.choice([1, 2, 3]),
        )

        image = self.single2uint(image)
        return image

    def add_Poisson_noise(self, img):
        img = np.clip((img * 255.0).round(), 0, 255) / 255.0
        vals = 10 ** (2 * random.random() + 2.0)  # [2, 4]
        if random.random() < 0.5:
            img = np.random.poisson(img * vals).astype(np.float32) / vals
        else:
            img_gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
            img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.0
            noise_gray = (
                np.random.poisson(img_gray * vals).astype(np.float32) / vals - img_gray
            )
            img += noise_gray[:, :, np.newaxis]
        img = np.clip(img, 0.0, 1.0)
        return img
    def add_Gaussian_noise(self, img, noise_level1=2, noise_level2=25):
        noise_level = random.randint(noise_level1, noise_level2)
        rnum = np.random.rand()
        if rnum > 0.6:  # add color Gaussian noise
            img += np.random.normal(0, noise_level / 255.0, img.shape).astype(
                np.float32
            )
        elif rnum < 0.4:  # add grayscale Gaussian noise
            img += np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(
                np.float32
            )
        else:  # add  noise
            L = noise_level2 / 255.0
            D = np.diag(np.random.rand(3))
            U = orth(np.random.rand(3, 3))
            conv = np.dot(np.dot(np.transpose(U), D), U)
            img += np.random.multivariate_normal(
                [0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]
            ).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        return img

    def add_JPEG_noise(self, img):
        quality_factor = random.randint(30, 95)
        img = cv2.cvtColor(self.single2uint(img), cv2.COLOR_RGB2BGR)
        result, encimg = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
        )
        img = cv2.imdecode(encimg, 1)
        img = cv2.cvtColor(self.uint2single(img), cv2.COLOR_BGR2RGB)
        return img

    def add_sharpening(self, img, weight=0.5, radius=50, threshold=10):
        if radius % 2 == 0:
            radius += 1
        blur = cv2.GaussianBlur(img, (radius, radius), 0)
        residual = img - blur
        mask = np.abs(residual) * 255 > threshold
        mask = mask.astype("float32")
        soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

        K = img + weight * residual
        K = np.clip(K, 0, 1)
        return soft_mask * K + (1 - soft_mask) * img

    def random_resizing1(self, image):
        h, w, c = image.shape

        updown_type = random.choices(self.updown_type, self.resize_prob)
        mode = random.choice(self.mode_list)

        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1

        if mode == "area":
            flags = cv2.INTER_AREA
        elif mode == "bilinear":
            flags = cv2.INTER_LINEAR
        elif mode == "bicubic":
            flags = cv2.INTER_CUBIC

        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=flags)
        image = cv2.resize(image, (w, h), interpolation=flags)
        image = np.clip(image, 0.0, 1.0)
        return image

    def random_resizing2(self, image):
        h, w, c = image.shape
        updown_type = random.choices(self.updown_type, self.resize_prob2)
        mode = random.choice(self.mode_list)

        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1

        if mode == "area":
            flags = cv2.INTER_AREA
        elif mode == "bilinear":
            flags = cv2.INTER_LINEAR
        elif mode == "bicubic":
            flags = cv2.INTER_CUBIC

        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=flags)
        image = cv2.resize(image, (w, h), interpolation=flags)
        image = np.clip(image, 0.0, 1.0)
        return image

    def generate_kernel1(self, image):
        kernel_size = random.choice(self.kernel_range)
        kernel = self.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            kernel_size,
            self.blur_sigma,
            self.blur_sigma,
            [-math.pi, math.pi],
            self.betag_range,
            self.betap_range,
            noise_range=None,
        )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        image = ndimage.filters.convolve(
            image, np.expand_dims(kernel, axis=2), mode="reflect"
        )
        return image.clip(min=0, max=255)

    def generate_kernel2(self, image):
        kernel_size = random.choice(self.kernel_range)
        kernel2 = self.random_mixed_kernels(
            self.kernel_list2,
            self.kernel_prob2,
            kernel_size,
            self.blur_sigma2,
            self.blur_sigma2,
            [-math.pi, math.pi],
            self.betag_range2,
            self.betap_range2,
            noise_range=None,
        )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        image = ndimage.filters.convolve(
            image, np.expand_dims(kernel, axis=2), mode="reflect"
        )
        return image.clip(min=0, max=255)

    def generate_sinc(self, image):
        sinc_kernel = self.pulse_tensor

        image = ndimage.filters.convolve(
            image, np.expand_dims(sinc_kernel, axis=2), mode="reflect"
        )
        return image.clip(min=0, max=255)

    def sigma_matrix2(self, sig_x, sig_y, theta):
        D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
        U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return np.dot(U, np.dot(D, U.T))

    def mesh_grid(self, kernel_size):
        ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = np.meshgrid(ax, ax)
        xy = np.hstack(
            (
                xx.reshape((kernel_size * kernel_size, 1)),
                yy.reshape(kernel_size * kernel_size, 1),
            )
        ).reshape(kernel_size, kernel_size, 2)
        return xy, xx, yy

    def pdf2(self, sigma_matrix, grid):
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
        return kernel

    def cdf2(D, grid):
        rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
        grid = np.dot(grid, D)
        cdf = rv.cdf(grid)
        return cdf

    def bivariate_Gaussian(
        self, kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
        if grid is None:
            grid, _, _ = self.mesh_grid(kernel_size)
        if isotropic:
            sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]])
        else:
            sigma_matrix = self.sigma_matrix2(sig_x, sig_y, theta)
        kernel = self.pdf2(sigma_matrix, grid)
        kernel = kernel / np.sum(kernel)
        return kernel

    def bivariate_generalized_Gaussian(
        self, kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True
    ):
        if grid is None:
            grid, _, _ = self.mesh_grid(kernel_size)
        if isotropic:
            sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]])
        else:
            sigma_matrix = self.sigma_matrix2(sig_x, sig_y, theta)
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.exp(
            -0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta)
        )
        kernel = kernel / np.sum(kernel)
        return kernel

    def bivariate_plateau(
        self, kernel_size, sig_x, sig_y, theta, beta, grid=None, isotropic=True
    ):
        if grid is None:
            grid, _, _ = self.mesh_grid(kernel_size)
        if isotropic:
            sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]])
        else:
            sigma_matrix = self.sigma_matrix2(sig_x, sig_y, theta)
        inverse_sigma = np.linalg.inv(sigma_matrix)
        kernel = np.reciprocal(
            np.power(np.sum(np.dot(grid, inverse_sigma) * grid, 2), beta) + 1
        )
        kernel = kernel / np.sum(kernel)
        return kernel

    def random_bivariate_Gaussian(
        self,
        kernel_size,
        sigma_x_range,
        sigma_y_range,
        rotation_range,
        noise_range=None,
        isotropic=True,
    ):
        assert kernel_size % 2 == 1, "Kernel size must be an odd number."
        assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        if isotropic is False:
            assert sigma_y_range[0] < sigma_y_range[1], "Wrong sigma_y_range."
            assert rotation_range[0] < rotation_range[1], "Wrong rotation_range."
            sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
            rotation = np.random.uniform(rotation_range[0], rotation_range[1])
        else:
            sigma_y = sigma_x
            rotation = 0

        kernel = self.bivariate_Gaussian(
            kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic
        )

        # add multiplicative noise
        if noise_range is not None:
            assert noise_range[0] < noise_range[1], "Wrong noise range."
            noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
            kernel = kernel * noise
        kernel = kernel / np.sum(kernel)
        return kernel

    def random_bivariate_generalized_Gaussian(
        self,
        kernel_size,
        sigma_x_range,
        sigma_y_range,
        rotation_range,
        beta_range,
        noise_range=None,
        isotropic=True,
    ):
        assert kernel_size % 2 == 1, "Kernel size must be an odd number."
        assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        if isotropic is False:
            assert sigma_y_range[0] < sigma_y_range[1], "Wrong sigma_y_range."
            assert rotation_range[0] < rotation_range[1], "Wrong rotation_range."
            sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
            rotation = np.random.uniform(rotation_range[0], rotation_range[1])
        else:
            sigma_y = sigma_x
            rotation = 0

        # assume beta_range[0] < 1 < beta_range[1]
        if np.random.uniform() < 0.5:
            beta = np.random.uniform(beta_range[0], 1)
        else:
            beta = np.random.uniform(1, beta_range[1])

        kernel = self.bivariate_generalized_Gaussian(
            kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic
        )

        # add multiplicative noise
        if noise_range is not None:
            assert noise_range[0] < noise_range[1], "Wrong noise range."
            noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
            kernel = kernel * noise
        kernel = kernel / np.sum(kernel)
        return kernel

    def random_bivariate_plateau(
        self,
        kernel_size,
        sigma_x_range,
        sigma_y_range,
        rotation_range,
        beta_range,
        noise_range=None,
        isotropic=True,
    ):
        assert kernel_size % 2 == 1, "Kernel size must be an odd number."
        assert sigma_x_range[0] < sigma_x_range[1], "Wrong sigma_x_range."
        sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
        if isotropic is False:
            assert sigma_y_range[0] < sigma_y_range[1], "Wrong sigma_y_range."
            assert rotation_range[0] < rotation_range[1], "Wrong rotation_range."
            sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
            rotation = np.random.uniform(rotation_range[0], rotation_range[1])
        else:
            sigma_y = sigma_x
            rotation = 0

        # TODO: this may be not proper
        if np.random.uniform() < 0.5:
            beta = np.random.uniform(beta_range[0], 1)
        else:
            beta = np.random.uniform(1, beta_range[1])

        kernel = self.bivariate_plateau(
            kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=isotropic
        )
        # add multiplicative noise
        if noise_range is not None:
            assert noise_range[0] < noise_range[1], "Wrong noise range."
            noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
            kernel = kernel * noise
        kernel = kernel / np.sum(kernel)

        return kernel

    def random_mixed_kernels(
        self,
        kernel_list,
        kernel_prob,
        kernel_size=21,
        sigma_x_range=[0.6, 5],
        sigma_y_range=[0.6, 5],
        rotation_range=[-math.pi, math.pi],
        betag_range=[0.5, 8],
        betap_range=[0.5, 8],
        noise_range=None,
    ):
        kernel_type = random.choices(kernel_list, kernel_prob)[0]
        if kernel_type == "iso":
            kernel = self.random_bivariate_Gaussian(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                noise_range=noise_range,
                isotropic=True,
            )
        elif kernel_type == "aniso":
            kernel = self.random_bivariate_Gaussian(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                noise_range=noise_range,
                isotropic=False,
            )
        elif kernel_type == "generalized_iso":
            kernel = self.random_bivariate_generalized_Gaussian(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                betag_range,
                noise_range=noise_range,
                isotropic=True,
            )
        elif kernel_type == "generalized_aniso":
            kernel = self.random_bivariate_generalized_Gaussian(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                betag_range,
                noise_range=noise_range,
                isotropic=False,
            )
        elif kernel_type == "plateau_iso":
            kernel = self.random_bivariate_plateau(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                betap_range,
                noise_range=None,
                isotropic=True,
            )

        elif kernel_type == "plateau_aniso":
            kernel = self.random_bivariate_plateau(
                kernel_size,
                sigma_x_range,
                sigma_y_range,
                rotation_range,
                betap_range,
                noise_range=None,
                isotropic=False,
            )
        return kernel

    # np.seterr(divide='ignore', invalid='ignore')

    def circular_lowpass_kernel(self, cutoff, kernel_size, pad_to=0):
        assert kernel_size % 2 == 1, "Kernel size must be an odd number."
        kernel = np.fromfunction(
            lambda x, y: cutoff
            * special.j1(
                cutoff
                * np.sqrt(
                    (x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2
                )
            )
            / (
                2
                * np.pi
                * np.sqrt(
                    (x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2
                )
            ),
            [kernel_size, kernel_size],
        )
        kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff ** 2 / (
            4 * np.pi
        )
        kernel = kernel / np.sum(kernel)
        if pad_to > kernel_size:
            pad_size = (pad_to - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        return kernel


# dataset.py
class Dataset(object):
    def __init__(self, images_dir, image_size, upscale_factor, degrad_param):
        deg = Degradation(dparam=degrad_param)
        self.filenames = [
            os.path.join(images_dir, x)
            for x in os.listdir(images_dir)
            if check_image_file(x)
        ]
        self.lr_transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(deg.degradation_pipeline),
                transforms.ToTensor(),
            ]
        )
        self.hr_transforms = transforms.Compose(
            [
                transforms.RandomCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        hr = self.hr_transforms(Image.open(self.filenames[idx]).convert("RGB"))
        lr = self.lr_transforms(hr)
        return lr, hr

    def __len__(self):
        return len(self.filenames)

def net_trainer(train_dataloader, eval_dataloader, model, pixel_criterion, net_optimizer, epoch, best_psnr, scaler, device, in_outputs_dir):
        model.train()
        losses = AverageMeter(name="PSNR Loss", fmt=":.6f")
        psnr = AverageMeter(name="PSNR", fmt=":.6f")
        
        """  Training Epoch Starting """
        for i, (lr, hr) in enumerate(train_dataloader):
            lr = lr.to(device)
            hr = hr.to(device)

            net_optimizer.zero_grad()

            with amp.autocast():
                preds = model(lr)
                loss = pixel_criterion(preds, hr)

            """ Scaler Update """
            scaler.scale(loss).backward()
            scaler.step(net_optimizer)
            scaler.update()

            """ Loss Update """
            losses.update(loss.item(), len(lr))
        
        """ 1 epoch , Update """
        print('losses.avg : ' + str(losses.avg) + ' epoch : ' + str(epoch))

        """  Test Epoch Starting """
        model.eval()
        for i, (lr, hr) in enumerate(eval_dataloader):
            lr = lr.to(device)
            hr = hr.to(device)
            with torch.no_grad():
                preds = model(lr)
            psnr.update(calc_psnr(preds, hr), len(lr))
    
        print('psnr.avg : ' + str(psnr.avg) + ' epoch : ' + str(epoch))

def gan_trainer(train_dataloader, eval_dataloader, generator, discriminator, pixel_criterion, content_criterion, adversarial_criterion, generator_optimizer, discriminator_optimizer, epoch, scaler, device, in_checkpoint_interval, in_outputs_dir):

    generator.train()
    discriminator.train()

    """ Losses average meter Setting """
    d_losses = AverageMeter(name="D Loss", fmt=":.6f")
    g_losses = AverageMeter(name="G Loss", fmt=":.6f")
    pixel_losses = AverageMeter(name="Pixel Loss", fmt=":6.4f")
    content_losses = AverageMeter(name="Content Loss", fmt=":6.4f")
    adversarial_losses = AverageMeter(name="adversarial losses", fmt=":6.4f")

    """ Model Evaluation measurements Setting """
    psnr = AverageMeter(name="PSNR", fmt=":.6f")

    # save_epoch
    checkpoint_interval = in_checkpoint_interval

    """  Training Epoch Starting """
    for i, (lr, hr) in enumerate(train_dataloader):
        """LR & HR Device Setting """
        lr = lr.to(device)
        hr = hr.to(device)

        """ Discriminator Optimization Initialization """
        discriminator_optimizer.zero_grad()

        with amp.autocast():
            """ Prediction """
            preds = generator(lr)
            """ After discriminator, loss calculation """
            real_output = discriminator(hr)
            d_loss_real = adversarial_criterion(real_output, True)

            fake_output = discriminator(preds.detach())
            d_loss_fake = adversarial_criterion(fake_output, False)

            d_loss = (d_loss_real + d_loss_fake) / 2

        """ Update weights """
        scaler.scale(d_loss).backward()
        scaler.step(discriminator_optimizer)
        scaler.update()

        """ generator optimization initialization"""
        generator_optimizer.zero_grad()

        with amp.autocast():
            """ prediction """
            preds = generator(lr)
            """ after generator, loss calculation """
            real_output = discriminator(hr.detach())
            fake_output = discriminator(preds)
            pixel_loss = pixel_criterion(preds, hr.detach())
            content_loss = content_criterion(preds, hr.detach())
            adversarial_loss = adversarial_criterion(fake_output, True)
            g_loss = 1 * pixel_loss + 1 * content_loss + 0.1 * adversarial_loss

        """ Update weights """
        scaler.scale(g_loss).backward()
        scaler.step(generator_optimizer)
        scaler.update()

        """ initialize generator """
        generator.zero_grad()

        """ loss update """
        d_losses.update(d_loss.item(), lr.size(0))
        g_losses.update(g_loss.item(), lr.size(0))
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))

    """ scheduler update """
    #discriminator_scheduler.step()
    #generator_scheduler.step()

    """ 1 epoch , update """
    print('d_losses.avg : ' + str(d_losses.avg) + ' epoch : ' + str(epoch))
    print('g_loddes.avg : ' + str(g_losses.avg) + ' epoch : ' + str(epoch))
    print('pixel_losses.avg : ' + str(pixel_losses.avg) + ' epoch : ' + str(epoch))
    print('content_losses.avg : ' + str(content_losses.avg) + ' epoch : ' + str(epoch))
    print('adversarial_losses.avg : ' + str(adversarial_losses.avg) + ' epoch : ' + str(epoch))

    """  test Epoch starting"""
    # generator.eval()
    # with torch.no_grad():
    #     for i, (lr, hr) in enumerate(eval_dataloader):
    #         lr = lr.to(device)
    #         hr = hr.to(device)
    #         preds = generator(lr)

    #         preds = preds.type(hr.type()).to(device)

    #         psnr.update(calc_psnr(preds, hr), len(lr))

    """ 1 epoch , update tensorboard """
    print('psnr (test) : ' + str(psnr.avg) + ' epoch : ' + str(epoch))

    model_script = torch.jit.trace(generator, lr)
    torch.jit.save(model_script, f"{in_outputs_dir}\\{epoch}.pth")

    torch.save(
            generator.state_dict(),
            os.path.join(in_outputs_dir, "model.pth"),
    )

def RecipeRun():

    # parameters
    ##################################
    param_train_datasets = 'D:\\workspace\\data\\img_align_celeba_256\\'
    param_valid_datasets = 'D:\\workspace\\data\\img_align_celeba_256\\'
    param_savepath = 'D:\\Model_Inference\\save_model\\real_esrgan\\'
    param_scale = 4
    param_resume_net = 'NONE'
    param_gpus = 2

    param_net_epochs = 1
    param_gan_epochs = 100
    param_checkpoint_interval = 1
    
    param_psnr_lr = 0.0001
    param_gan_lr = 0.0002

    param_patch_size = 256
    param_batch_size = 4

    param_resume_d = 'NONE'
    param_resume_g = 'NONE'

    # for degradation.......
    degrad_param = degradation_params()
    ###################################

    """ for saving,  weight path setting """ 
    in_outputs_dir = param_savepath
    if not os.path.exists(in_outputs_dir):
        os.makedirs(in_outputs_dir)

    """ GPU device setting"""
    cudnn.benchmark = True
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    """ Torch Seed setting """
    seed_num = 123
    torch.manual_seed(seed_num)

    """ RealESRGAN psnr model setting """
    generator = Generator(param_scale).to(device)

    pixel_criterion = nn.L1Loss().to(device)
    net_optimizer = torch.optim.Adam(generator.parameters(), param_psnr_lr, (0.9, 0.99))
    interval_epoch = math.ceil(param_net_epochs // 8)
    epoch_indices = [interval_epoch, interval_epoch * 2, interval_epoch * 4, interval_epoch * 6]
    net_scheduler = torch.optim.lr_scheduler.MultiStepLR(net_optimizer, milestones=epoch_indices, gamma=0.5)
    scaler = amp.GradScaler()

    total_net_epoch = param_net_epochs
    start_net_epoch = 0
    best_psnr = 0

    """ RealESNet checkpoint model weight loading """
    if os.path.exists(param_resume_net):
        checkpoint = torch.load(param_resume_net)
        generator.load_state_dict(checkpoint['model_state_dict'])
        net_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_net_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        best_psnr = checkpoint['best_psnr']

    """ RealESRNet log info printing """
    print(
                f"RealESRNET MODEL INFO:\n"
                f"\tScale factor:                  {param_scale}\n"

                f"RealESRGAN TRAINING INFO:\n"
                f"\tTotal Epoch:                   {param_net_epochs}\n"
                f"\tStart Epoch:                   {start_net_epoch}\n"
                f"\tTrain directory path:          {param_train_datasets}\n"
                f"\tTest directory path:           {param_valid_datasets}\n"
                f"\tOutput weights directory path: {param_savepath}\n"
                f"\tPSNR learning rate:            {param_psnr_lr}\n"
                f"\tPatch size:                    {param_patch_size}\n"
                f"\tBatch size:                    {param_batch_size}\n"
                )

    """ dataset, dataset configuration"""
    train_dataset = Dataset(param_train_datasets, param_patch_size, param_scale, degrad_param)
    train_dataloader = DataLoader(
                            dataset=train_dataset,
                            batch_size=param_batch_size,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True
                        )
    eval_dataset = Dataset(param_valid_datasets, param_patch_size, param_scale, degrad_param)
    eval_dataloader = DataLoader(
                                dataset=eval_dataset, 
                                batch_size=1,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True
                                )
    """NET Training"""
    print('start_net_epoch, total_net_epoch : %d, %d'%(start_net_epoch, total_net_epoch))
    for epoch in range(start_net_epoch, total_net_epoch):
        print('NET :: epoch %d'%epoch)
        net_trainer(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, model=generator, pixel_criterion=pixel_criterion, net_optimizer=net_optimizer, epoch=epoch, best_psnr=best_psnr, scaler=scaler, device=device, in_outputs_dir=in_outputs_dir)
        net_scheduler.step()


    """ RealESNet checkpoint weight loading """
    discriminator = Discriminator().to(device)

    total_gan_epoch = param_gan_epochs
    start_gan_epoch = 0

    content_criterion = VGGLoss().to(device)
    adversarial_criterion = GANLoss().to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=param_gan_lr, betas=(0.9, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=param_gan_lr, betas=(0.9, 0.999))

    interval_epoch = math.ceil(param_gan_epochs // 8)
    epoch_indices = [interval_epoch, interval_epoch * 2, interval_epoch * 4, interval_epoch * 6]

    discriminator_scheduler = torch.optim.lr_scheduler.MultiStepLR(discriminator_optimizer, milestones=epoch_indices, gamma=0.5)
    generator_scheduler = torch.optim.lr_scheduler.MultiStepLR(generator_optimizer, milestones=epoch_indices, gamma=0.5)

    """ checkpoint weight loading """
    if os.path.exists(param_resume_g) :
        """ resume generator """
        checkpoint_g = torch.load(param_resume_g)
        generator.load_state_dict(checkpoint_g['model_state_dict'])
        start_gan_epoch = checkpoint_g['epoch'] + 1
        generator_optimizer.load_state_dict(checkpoint_g['optimizer_state_dict'])

    if os.path.exists(param_resume_d):
        """ resume discriminator """
        checkpoint_d = torch.load(param_resume_d)
        discriminator.load_state_dict(checkpoint_d['model_state_dict'])
        discriminator_optimizer.load_state_dict(checkpoint_d['optimizer_state_dict'])

    """ RealESGAN log info printing """
    print(
                f"RealESRGAN MODEL INFO:\n"
                f"\tScale factor:                  {param_scale}\n"

                f"RealESRGAN TRAINING INFO:\n"
                f"\tTotal Epoch:                   {param_gan_epochs}\n"
                f"\tStart Epoch:                   {start_gan_epoch}\n"
                f"\tTrain directory path:          {param_train_datasets}\n"
                f"\tTest directory path:           {param_valid_datasets}\n"
                f"\tOutput weights directory path: {param_savepath}\n"
                f"\tPSNR learning rate:            {param_psnr_lr}\n"
                f"\tPatch size:                    {param_patch_size}\n"
                f"\tBatch size:                    {param_batch_size}\n"
                f"\tCheckpoint_iterval:            {param_checkpoint_interval}\n"
                )

    """GAN Training"""
    print('start_gan_epoch, total_gan_epoch : %d, %d'%(start_gan_epoch, total_gan_epoch))
    for epoch in range(start_gan_epoch, total_gan_epoch):
        print('GAN epoch : %d'%epoch)
        gan_trainer(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, generator=generator, discriminator=discriminator, pixel_criterion=pixel_criterion, content_criterion=content_criterion, adversarial_criterion=adversarial_criterion, generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, epoch=epoch, scaler=scaler, device=device, in_checkpoint_interval=param_checkpoint_interval, in_outputs_dir=in_outputs_dir)
        discriminator_scheduler.step()
        generator_scheduler.step()

if __name__ == '__main__':

    RecipeRun()

    print('complete....')

