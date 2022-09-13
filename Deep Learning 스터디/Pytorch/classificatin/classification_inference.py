import torch
import torchvision.transforms as transforms

class ModelHandler:
    def __init__(self, model_path, resize = 32):
        self.__resize = resize
        self.__device = "cpu"

        self.__model_file_path = model_path
        self.__model, self.__labels = self.__load_model()

    def __call__(self, data):
        output = self.__inference(data)
        return output

    def __load_model(self):
        labels_info = {'labels' : ''}
        model = torch.jit.load(self.__model_file_path, map_location=self.__device, _extra_files=labels_info)
        model.eval()

        label_list = labels_info['labels'].decode('ascii').split("#")

        return model, label_list

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
        output = dict()
        for score, label in zip(result[0][0], self.__labels):
            output[label] = score.item()

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

    print(*list(inference_result.items()), sep="\n")

    v = list(inference_result.values())
    k = list(inference_result.keys())

    print(k[v.index(max(v))], max(v))