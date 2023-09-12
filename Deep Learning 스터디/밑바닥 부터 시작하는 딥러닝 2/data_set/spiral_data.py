import numpy as np
import matplotlib.pyplot as plt

def load_spiral_data():
    N = 100  # 클래스당 샘플 수
    DIM = 2  # 데어터 요소 수
    CLS_NUM = 3  # 클래스 수

    x = np.zeros((N*CLS_NUM, DIM))
    t = np.zeros((N*CLS_NUM, CLS_NUM), dtype=np.int16)

    for j in range(CLS_NUM):
        for i in range(N): # N*j, N*(j+1)):
            rate = i / N
            radius = 1.0*rate
            theta = j*4.0 + 4.0*rate + np.random.randn()*0.2

            ix = N*j + i
            x[ix] = np.array([radius*np.sin(theta),
                              radius*np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t

def show_data(data):
    data_x = data[:, 0]
    data_y = data[:, 1]

    plt.scatter(data_x[:100], data_y[:100], c='#33FFCE')
    plt.scatter(data_x[100:200], data_y[100:200], c='#FF5733')
    plt.scatter(data_x[200:], data_y[200:], c='#5733FF')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()

if __name__ == "__main__":
    data, t = load_spiral_data()
    show_data(data)    

    