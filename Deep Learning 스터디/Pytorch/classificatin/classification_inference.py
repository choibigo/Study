import json
import os
import pickle
import torch
import torchvision.transforms as transforms
import grpc
import mpp
from google.protobuf import wrappers_pb2

class ModelHandler:
    def __init__(self, model_path, resize = 32):
        self.__resize = resize
        self.__device = "cpu"

        self.__model_file_path = model_path
        self.__model = self.__load_model()

    def __call__(self, data):
        output = self.__inference(data)
        return output

    def __load_model(self):
        model = torch.jit.load(self.__model_file_path, map_location=self.__device)
        model.eval()
        return model

    def __inference(self, data):
        data = self.__preprocessing(data)
        output = self.__model(data)
        output = self.__postprocessing(output)
        return output

    def __preprocessing(self, image):
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((self.__resize,self.__resize))])
        image = transform(image)
        image = image.unsqueeze(0)
        return image

    def __postprocessing(self, result):
        output = [score.item() for score in result[0][0]]
        return output

if __name__ == "__main__":
    
    import cv2
    test_data_path = r"D:\workspace\data\Classification\CIFAR-10-images-master\train\ship\0000.jpg"
    test_data = cv2.imread(test_data_path, 1)

    model_path = r"D:\temp\save_model\GoogleNet\googlenet_45.pth"

    # # =================================================================== #

    handler = ModelHandler(model_path)
    inference_result =  handler(test_data)

    # # =================================================================== #

    print(inference_result)
    print(inference_result.index(max(inference_result)))
   
