import sys
sys.path.append('..')
import numpy as np
from common.optimizer import SGD
from data_set.spiral_data import load_spiral_data
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

def show_loss(loss_list):
    plt.plot(np.arange(len(loss_list)), loss_list, label='train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":

    # config
    total_epoch = 300
    batch_size = 30
    hidden_size = 10
    learning_rate = 1.0

    # Data load
    x, t = load_spiral_data()
    model = TwoLayerNet(input_size=2,hidden_size=hidden_size, output_size=3)
    # 입력 차원수=2, hidden_size, class 분류 개수 = 3

    # Optimizer
    optimizer = SGD(lr=learning_rate)

    data_size = len(x)
    max_iters = data_size // batch_size
    total_loss = 0
    loss_list = []

    for epoch in range(total_epoch):

        # data size개 만큼 random한 index를 가져와서 suffle 을 구현 했음
        idx = np.random.permutation(data_size)
        x = x[idx]
        t = t[idx]

        for iters in range(max_iters):
            batch_x = x[iters*batch_size:(iters+1)*batch_size] # batch_size 만큼씩 잘라서 batch 데이터를 만든다.
            batch_t = t[iters*batch_size:(iters+1)*batch_size] # batch_size 만큼씩 잘라서 batch 데이터를 만든다.

            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)

            total_loss += loss

        loss_list.append(total_loss/max_iters)
        total_loss = 0
        print(f"Epoch: {epoch} END | Loss : {loss}")

    
    # show_loss(loss_list)

    # Inference
    test_x, label_t = load_spiral_data()
    predict = model.predict(test_x)

    predict = np.argmax(predict, axis=1)
    label = np.argmax(label_t, axis=1)

    compare = np.equal(predict, label)
    print(f"Predict Acc: {sum(compare) / len(test_x)}")

    h = 0.001
    x_min, x_max = test_x[:, 0].min() - .1, test_x[:, 0].max() + .1
    y_min, y_max = test_x[:, 1].min() - .1, test_x[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X = np.c_[xx.ravel(), yy.ravel()]
    score = model.predict(X)
    predict_cls = np.argmax(score, axis=1)
    Z = predict_cls.reshape(xx.shape)
    plt.contourf(xx, yy, Z)
    plt.axis('off')

    for i in range(len(test_x)):
        if predict[i] == 0:
            plt.scatter([test_x[i][0]], [test_x[i][1]], marker="s", c='#33FFCE')
        elif predict[i] == 1:
            plt.scatter([test_x[i][0]], [test_x[i][1]], marker="^", c='#FF5733')
        elif predict[i] == 2:
            plt.scatter([test_x[i][0]], [test_x[i][1]], marker="o", c='#5733FF')
    plt.show()