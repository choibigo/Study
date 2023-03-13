import os
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

    def __call__(self, image):
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.unsqueeze(torch.from_numpy(image), 0).to('cpu')

        predict_list = self.net(image_tensor)
        
        side_out_1, side_out_2, side_out_3, side_out_4, side_out_5, fuse = predict_list
        side_out_1 = self.__min_max_normalize(side_out_1)
        side_out_2 = self.__min_max_normalize(side_out_2)
        side_out_3 = self.__min_max_normalize(side_out_3)
        side_out_4 = self.__min_max_normalize(side_out_4)
        side_out_5 = self.__min_max_normalize(side_out_5)
        fuse = self.__min_max_normalize(fuse)

        return side_out_1, side_out_2, side_out_3, side_out_4, side_out_5, fuse

    def __min_max_normalize(self, input_data):
        output = torch.squeeze(input_data, 0).to('cpu')
        output = output.detach().data.numpy().transpose(1,2,0)

        min_value =  np.min(output)
        max_value =  np.max(output)
        output = ((output - min_value) / (max_value - min_value)) * 255
        output = output.astype(np.uint8)

        return output

if __name__ == "__main__":
    epoch = "0"
    handler = Handler(f"D:\\hed_out_dir\\epoch-{epoch}-checkpoint.pt")
    input_path = r"D:\workspace\data\BSDS\train\aug_data\0.0_1_0\3096.jpg"
    output = handler(cv2.imread(input_path))


    # test_root= r"D:\workspace\data\BSDS\train\aug_data\0.0_1_0"
    # gt_root= r"D:\workspace\data\BSDS\train\aug_gt\0.0_1_0"
    # save_root = f"D:\\hed_out_dir\\inference_{epoch}"

    # for file_name in os.listdir(test_root):
    #     name, ext = os.path.splitext(file_name)
    #     os.makedirs(os.path.join(save_root, name), exist_ok=True)
    #     input_image = cv2.imread(os.path.join(test_root, file_name))
    #     edge_image = cv2.imread(os.path.join(gt_root, f"{name}.png"))
    #     predict_list = handler(input_image)

    #     cv2.imwrite(os.path.join(save_root, name, f'input_image.png'), input_image)
    #     cv2.imwrite(os.path.join(save_root, name, f'input_gt_image.png'), edge_image)
    #     cv2.imwrite(os.path.join(save_root, name, f'side_output_1.png'), predict_list[0])
    #     cv2.imwrite(os.path.join(save_root, name, f'side_output_2.png'), predict_list[1])
    #     cv2.imwrite(os.path.join(save_root, name, f'side_output_3.png'), predict_list[2])
    #     cv2.imwrite(os.path.join(save_root, name, f'side_output_4.png'), predict_list[3])
    #     cv2.imwrite(os.path.join(save_root, name, f'side_output_5.png'), predict_list[4])
    #     cv2.imwrite(os.path.join(save_root, name, f'fuse.png'), predict_list[5])
        