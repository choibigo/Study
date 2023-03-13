import os
import numpy as np
import cv2
import torchmetrics
import torch

def cal_average_precision(gt_image, predict_image):
    gt_tensor = torch.from_numpy(gt_image)
    predict_tensor = torch.from_numpy(predict_image)

    average_precision = torchmetrics.AveragePrecision(task="binary")
    return  average_precision(predict_tensor, gt_tensor)

if __name__ == "__main__":
    epoch = "34"
    for num in os.listdir(r'D:\hed_out_dir\inference_0'):
        input_root = f"D:\\hed_out_dir\\inference_{epoch}\\{num}"
        gt_image_path = os.path.join(input_root, "input_gt_image.png")
        side_output_1_path = os.path.join(input_root, "side_output_1.png")
        side_output_2_path = os.path.join(input_root, "side_output_2.png")
        side_output_3_path = os.path.join(input_root, "side_output_3.png")
        side_output_4_path = os.path.join(input_root, "side_output_4.png")
        side_output_5_path = os.path.join(input_root, "side_output_5.png")
        fuse_image_path = os.path.join(input_root, "fuse.png")

        gt_image = cv2.threshold(cv2.imread(gt_image_path), 0, 1, cv2.THRESH_BINARY)[1].astype(np.float64)
        side_output_1 = cv2.imread(side_output_1_path).astype(np.float64) / 255.0
        side_output_2 = cv2.imread(side_output_2_path).astype(np.float64) / 255.0
        side_output_3 = cv2.imread(side_output_3_path).astype(np.float64) / 255.0
        side_output_4 = cv2.imread(side_output_4_path).astype(np.float64) / 255.0
        side_output_5 = cv2.imread(side_output_5_path).astype(np.float64) / 255.0
        fuse_image = cv2.imread(fuse_image_path).astype(np.float64) / 255.0

        side_output_1_ap = cal_average_precision(gt_image, side_output_1)
        side_output_2_ap = cal_average_precision(gt_image, side_output_2)
        side_output_3_ap = cal_average_precision(gt_image, side_output_3)
        side_output_4_ap = cal_average_precision(gt_image, side_output_4)
        side_output_5_ap = cal_average_precision(gt_image, side_output_5)
        fuse_ap = cal_average_precision(gt_image, fuse_image)

        print(f"side_output_1_ap: {side_output_1_ap}")
        print(f"side_output_2_ap: {side_output_2_ap}")
        print(f"side_output_3_ap: {side_output_3_ap}")
        print(f"side_output_4_ap: {side_output_4_ap}")
        print(f"side_output_5_ap: {side_output_5_ap}")
        print(f"fuse_ap: {fuse_ap}")

        with open(os.path.join(input_root, 'average_precision.txt'), 'w') as f:
            f.write(f"side_output_1_ap: {side_output_1_ap} \n")
            f.write(f"side_output_2_ap: {side_output_2_ap} \n")
            f.write(f"side_output_3_ap: {side_output_3_ap} \n")
            f.write(f"side_output_4_ap: {side_output_4_ap} \n")
            f.write(f"side_output_5_ap: {side_output_5_ap} \n")
            f.write(f"fuse_ap: {fuse_ap}")
