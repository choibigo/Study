import cv2
import os
from glob import glob
import random

import torch
from torchvision import datasets as dset
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

from siamesenet import SiameseNet
from arguments import get_config

if __name__ == "__main__":
    transformer = transforms.Compose([
                        transforms.Resize(105),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=0.5,std=0.5)])

    test_data = dset.MNIST(root='MNIST_data/',train=False,transform=transformer, download=True)

    # region Test
    # Check Test Data index 0
    # test_image, test_label = test_data[0]

    # plt.imshow(test_image.squeeze().numpy(), cmap='gray')
    # plt.title('%i' % test_label)
    # plt.show()

    # print(test_image.size())
    # print('number of test data:', len(test_data))


    # endregion Test

    class MNISTTest(Dataset):
        def __init__(self, dataset,trial):
            self.dataset = dataset
            self.trial = trial
            if trial > 950:
                self.trial = 950

        def __len__(self):
            return self.trial * 10

        def __getitem__(self, index):
            share, remain = divmod(index,10)
            label = (share//10)%10
            image1 = self.dataset[label][share][0]
            image2 = self.dataset[remain][random.randrange(len(self.dataset[remain]))][0]

            return image1, image2, label
    

    image_by_num = [[],[],[],[],[],[],[],[],[],[]]
    for x,y in test_data:
        image_by_num[y].append(x)

    test_data1 = MNISTTest(image_by_num,trial=950) #MAX trial = 950
    test_loader = DataLoader(test_data1, batch_size=10)

    config = get_config()
    config.num_model = "1"
    config.logs_dir = "./result/1"
    model = SiameseNet()
    is_best = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = r'D:\workspace\Difficult\my_repository\Deep Learning 스터디\논문읽기\논문 구현\siamese\result\1\models\model_ckpt_1.pt'

    ckpt = torch.load(model_path)

    model.load_state_dict(ckpt['model_state'])
    model.to(device) 
    print(f"[*] Load model {os.path.basename(model_path)}, best accuracy {ckpt['best_valid_acc']}")

    correct_sum = 0
    num_test = len(test_loader) 
    print(f"[*] Test on {num_test} pairs.")

    for i, (x1, x2, y) in enumerate(test_loader):
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        out = model(x1, x2)

        # if y[0].data == 1:
        #     for iii in range(10):
        #         x1_image = x1[iii].cpu().squeeze().numpy()
        #         cv2.imwrite(f".//debug_image//x1_inference//{iii}_x1_image.png", x1_image*255)

        #     for iii in range(10):
        #         x2_image = x2[iii].cpu().squeeze().numpy()
        #         cv2.imwrite(f".//debug_image//x2_inferenece//{iii}_x2_image.png", x2_image*255)

        #     print()

        y_pred_score = torch.sigmoid(out)
        y_pred_index = torch.argmax(y_pred_score)

        # print(f"Out : {out}, Score : {y_pred_score}, Max Index : {y_pred_index}")

        if y_pred_index == y[0].item():
            correct_sum += 1

        # plt.imshow(x1.cpu().squeeze().numpy(), cmap='gray')
        # plt.imshow(x2.cpu().squeeze().numpy(), cmap='gray')
        # plt.show()

        if i % 100 == 0:
            print(f"{i}/{num_test} End")

    test_acc = (correct_sum / num_test) * 100

    print(f"test_acc : {test_acc}")



    print("Success")