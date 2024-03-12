import os
import numpy as np
import torch


data_root_path = r"D:\workspace\Difficult\inference_result\cogvlm\behavior\feature_2_27"

total_array = np.array([[]]).reshape(0, 8192)

for num in os.listdir(data_root_path):
    temp = torch.load(os.path.join(data_root_path, f"{num}", "layer_output.pt"))

    total_array = np.append(total_array, temp.type(torch.float32).cpu().numpy(), axis=0)


np.save('feature_2_27.npy', total_array)

