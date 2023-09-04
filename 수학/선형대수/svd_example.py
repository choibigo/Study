import os
import cv2
import scipy
import numpy as np
import matplotlib.pyplot as plt

debug_image_flag = False
sigma_show_flag = False
K = 30

def sigma_show(s):
    # sigma 배열 보기 x 축은 index, y축은 값
    # sigma 값 자체가 함수에서 분해되어져 나올때 크기순으로 sort되어 있음
    plt.plot(s)
    plt.axis([-10, len(s), 0, max(s)])
    plt.show()

def debug_image_show(**kwags):
    for title, image in kwags.items():
        cv2.imshow(title, image)

    cv2.waitKey(0)
    cv2.destroyWindow()

if "__main__" == __name__:
    load_image = scipy.datasets.face()

    load_image = cv2.imread(r'D:\test_image\cat1.jpg', 1)

    gray_image = cv2.cvtColor(load_image, cv2.COLOR_BGR2GRAY)
    input_image = gray_image/255

    U, s, Vt = np.linalg.svd(input_image)

    # print(U.shape) # left singular vector, Input Image의 row x row
    # print(s.shape) # 1차원의 배열이다 => singular value들로 구성되어 있고 이 배열을 통해 실제 sigma 배열을 만들 수 있다. Input image의 row 크기의 벡터
    # print(Vt.shape)# right singular vector, Input Image의 col x col

    sigma = np.zeros(input_image.shape)
    for i in range(len(s)):
        sigma[i,i] = s[i]
    # 1차원의 singular value들로 sigma 행렬 만들기

    save_folder_path = "./compression_image/"
    os.makedirs(save_folder_path)
    for i in range(0, K):
        compression_image = U @ sigma[:, :i] @ Vt[:i, :] # eigen value중 큰것 K개만 사용 sigma는 row x k, Vt는 k x col 형태로 만든다.
        cv2.imwrite(f"{save_folder_path}/{i}.png", compression_image*255)

    # 원래 이미지에서 재구성한 이미지 차이의 Norm값을 계산한다.
    diff1 = np.linalg.norm(input_image - U @ sigma @ Vt)
    print(f"Difference (Input - Reconstruction) = {diff1}")

    if sigma_show_flag:
        sigma_show(s)

    if debug_image_flag:
        debug_image_info ={
            'Input':input_image,
            'Reconstruction': U @ sigma @ Vt
        }
        debug_image_show(**debug_image_info)