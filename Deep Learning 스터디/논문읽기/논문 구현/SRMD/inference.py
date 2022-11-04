import os
import pickle
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from scipy.ndimage import convolve
from PIL import Image

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

class ModelHandler:
    def __init__(self, model_path):
        
        extra_files = {"kernel":None}
        self.__model = torch.jit.load(model_path, map_location="cpu", _extra_files=extra_files)
        self.__model.eval()
        self.__model = torch.jit.optimize_for_inference(torch.jit.script(self.__model))
        kernel  = pickle.loads(extra_files['kernel'])
        self.transform = transforms.Compose([
                                transforms.Resize((256, 256)),
                                transforms.Lambda(lambda x: self.__scaling(x)),
                                transforms.Lambda(lambda x: kernel.ConcatDegraInfo(x)),
                                transforms.ToTensor()
                        ])
                        
    def __call__(self, data):
        data = Image.fromarray(data)
        data_tensor = self.transform(data)
        data_tensor = torch.unsqueeze(data_tensor, dim=0)

        output = self.__model(data_tensor)

        output = torch.squeeze(output, dim=0)
        output = output.detach().data.numpy()
        output = output.transpose(1,2,0)
        min_value =  np.min(output)
        max_value =  np.max(output)
        output = ((output - min_value) / (max_value - min_value)) * 255
        output = output.astype(np.uint8)

        return output

    def __scaling(self, image):
        return np.array(image) / 255.0

if __name__ == "__main__":
    test_data_path = r"D:\workspace\data\super_resolution_patch\1.BMP_4_8.png"
    model_path = r"D:\Model_Inference\save_model\srmd\39.pth"
    test_data = cv2.imread(test_data_path, 1)

    # =================================================================== #

    handler = ModelHandler(model_path)
    inference_result =  handler(test_data)

    # =================================================================== #
    inference_result = cv2.cvtColor(inference_result, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(r"D:\Model_Inference\inference\SRMD\result.png", inference_result)

    cv2.imwrite(r"D:\Model_Inference\inference\SRMD\input.png", test_data)

    print("Success")