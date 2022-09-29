import cv2
import numpy as np
import torch
from torchvision import transforms

def eval(image_path):
    model = torch.jit.load(r"D:\Model_Inference\save_model\Fast-Scnn\48.pth", map_location='cpu').eval()

    transfom= transforms.Compose([
        transforms.ToTensor()
    ])

    image = cv2.imread(image_path, 1)
    image = transfom(image)
    image = torch.unsqueeze(image, dim=0)
    outputs = model(image)

    pred = torch.argmax(outputs[0], 1)
    pred = pred.cpu().data.numpy()
    predict = pred.squeeze(0)
    return predict.astype(np.uint8)

if __name__ == '__main__':
    result = eval(r"D:\workspace\data\ADI_BALL_768\image\1.png")

    from PIL import Image
    palette = [
        0, 0, 0,
        255, 255, 255
    ]
    mask = Image.fromarray(result.astype(np.uint8))
    mask.putpalette(palette)
    save_path= "D:\\Model_Inference\\inference\\Fast-SCNN\\fast_scnn_result.png"
    mask.save(save_path)