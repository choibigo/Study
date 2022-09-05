import json
import os
import pickle
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class ModelHandler:
    def __init__(self, data, context):
        self._initialize()
        model_config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
        with open(model_config_file_path, "r", encoding='utf-8') as config_file:
            self.config = json.load(config_file)
            self.model_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config['model_file'])

        if self._framework_type.lower() == 'pytorch':
            self.model = self._load_model()

        else:
            raise NotImplementedError

    def __call__(self, data, context):
        if self._framework_type.lower() == 'pytorch':
            return self.inference(data, context)
        else:
            raise NotImplementedError

    def _initialize(self):
        self._framework_type = "pytorch"
        self.device = 'cpu'

    def _load_model(self):
        model = torch.jit.load(self.model_file_path)
        model.eval().to(self.device)
        return model

    def inference(self, data, context, *args, **kwargs):
        data = pickle.loads(data)
        data = self._preprocessing(data)
        output = self.model(data)
        output = self._postprocessing(output)
        output = pickle.dumps(output) 
        
        return output, context

    def _preprocessing(self, image):
        image = Image.fromarray(np.uint8(image))
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((256, 256))])
        image = transform(image)
        image = image.unsqueeze(0)

        return image

    def _postprocessing(self, result):
        result = result.detach().numpy().astype(np.float32)
        output = (255*(result - np.min(result))/np.ptp(result)).astype(np.uint8)
        output = output.squeeze()
        output = output.transpose(1,2,0)
        return output


if __name__ == "__main__":
    
    import cv2
    from skimage.io import imsave

    test_data_path = r"D:\workspace\data\pix2pix\facades\test\b\cmp_b0349.png"
    test_data = cv2.imread(test_data_path, 1)
    test_data_pickle = pickle.dumps(test_data)

    # =================================================================== #

    handler = ModelHandler(None, None)
    inference_result, context =  handler(test_data_pickle, None)

    # =================================================================== #

    inference_result = pickle.loads(inference_result)
    imsave(r"D:\temp\inference\Pix2Pix\result.png", inference_result)
    print("Success")

