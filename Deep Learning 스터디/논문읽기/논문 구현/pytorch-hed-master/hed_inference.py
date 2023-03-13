import torch
import torch.nn as nn
from networks import HED
import numpy as np
import cv2

class Handler:
    def __init__(self, checkpoint_path):
        self.net = nn.DataParallel(HED('cuda'), device_ids=[0]).eval()
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['net'])

    def __call__(self, input_image_path):
        image = cv2.imread(input_image_path).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.unsqueeze(torch.from_numpy(image), 0).to('cpu')

        predict_list = self.net(image_tensor)

        output = predict_list[5]
        output = torch.squeeze(output, 0).to('cpu')
        output = output.detach().data.numpy().transpose(1,2,0)
        normalize_output = self.__min_max_normalize(output)
        
        return normalize_output

    def __min_max_normalize(self, data):
        min_value =  np.min(data)
        max_value =  np.max(data)
        output = ((data - min_value) / (max_value - min_value)) * 255
        output = output.astype(np.uint8)

        return output

if __name__ == "__main__":
    handler = Handler(r"D:\hed_out_dir\epoch-0-checkpoint.pt")

    output = handler(r"D:\temp\image_origin.png")

    cv2.imwrite(r"D:\temp\inference.png", output)

    print("Success")
