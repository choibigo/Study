import cv2
import numpy as np
import torch
from torchvision import transforms

iter = 7

def eval(image_path):
    model = torch.jit.load(f"D:\\Model_Inference\\save_model\\DeeplabV3\\{iter}.pth", map_location='cpu').eval()

    transfom= transforms.Compose([
        transforms.ToTensor()
    ])

    image = cv2.imread(image_path, 1)
    image = transfom(image)
    image = torch.unsqueeze(image, dim=0)
    outputs = model(image)

    pred = torch.argmax(outputs, 1)
    pred = pred.cpu().data.numpy()
    predict = pred.squeeze(0)
    return predict.astype(np.uint8)

if __name__ == '__main__':
    result = eval(r"D:\workspace\data\ADI_BALL\image\2.png")

    from PIL import Image
    palette = [
        44, 44, 44,
        255, 255, 255
    ]
    mask = Image.fromarray(result.astype(np.uint8))
    mask.putpalette(palette)
    save_path= f"D:\\Model_Inference\\inference\\DeeplabV3\\deeblabv3_{iter}.png"
    mask.save(save_path)

    print("Success")