import torch
import torchvision.transforms as transforms
import json

class ModelHandler:
    def __init__(self, model_path, resize = 32):
        self.__resize = resize
        self.__device = "cpu"

        self.__model_file_path = model_path
        self.__model, self.__label_info = self.__load_model()

    def __call__(self, data):
        output = self.__inference(data)
        return output

    def __load_model(self):
        extra_files = {'label_info' : {}}
        model = torch.jit.load(self.__model_file_path, map_location=self.__device, _extra_files=extra_files)
        model.eval()

        label_info = extra_files['label_info'].decode('ascii')
        label_info = json.loads(label_info)

        return model, label_info

    def __inference(self, data): 
        data = self.__preprocessing(data)
        output, _, _ = self.__model(data)
        output = self.__postprocessing(output[0])
        return output

    def __preprocessing(self, image):
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((self.__resize,self.__resize))])
        image = transform(image)
        image = image.unsqueeze(0)
        return image

    def __postprocessing(self, result):

        output = list()
        for i in range(self.__label_info['label_count']):
            current_label = self.__label_info[f'label_{i}']
            current_label['score'] = result[i].item()
            output.append(current_label)

        output = sorted(output, key = lambda x : x['score'], reverse=True)

        return output

if __name__ == "__main__":

    import cv2
    test_data_path = r"D:\workspace\data\Classification\CIFAR-10-images-master\train\ship\0000.jpg"
    test_data = cv2.imread(test_data_path, 1)

    model_path = r"D:\temp\save_model\GoogleNet\googlenet_1.pth"

    # # =================================================================== #

    handler = ModelHandler(model_path)
    inference_result =  handler(test_data)

    # # =================================================================== #

    print(*inference_result, sep="\n")
