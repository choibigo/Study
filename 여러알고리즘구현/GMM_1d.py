import random
import numpy as np
import matplotlib.pyplot as plt

def normpdf(xx, mu, var):
    return (1 / (np.sqrt(2 * np.pi * var))) * np.exp(-(xx - mu) ** 2 / (2 * var))

np.random.seed(1)

mu = np.array([0, 15])
vars = np.array([12, 3])
n_data = 1000
phi = np.array([0.7, 0.3])
X = []

for i_data in range(n_data):
    if np.random.rand() < phi[0]: # 70% 확률로 1번 분포에서 데이터를 뽑음
        X.append(np.random.randn() * np.sqrt(vars[0]) + mu[0])
    else: # 30% 확률로 2번 분포에서 데이터를 뽑음
        X.append(np.random.randn() * np.sqrt(vars[1]) + mu[1])

# 데이터 분포 확인
plt.figure(figsize=(10, 6))
xx = np.linspace(-30, 30, 100)
yy1 = normpdf(xx, mu[0], vars[0])
yy2 = normpdf(xx, mu[1], vars[1])
result_distribution = yy1 * phi[0] + yy2 * phi[1] # 1번 분포를 0.7, 2번 분포를 0.3을 곱해서 더해 최종 분포를 만들어 낸다.


est_mu = [-25, 20]
est_vars = [7, 9.5]

est_w = np.zeros((n_data, 2)) # 각 데이터가 1번 분포에 속할지 2번 분포에 속할지 저장하는 변수
est_phi = [0.5, 0.5]

# 초기 그래프
est_yy1 = normpdf(xx, est_mu[0], est_vars[0])
est_yy2 = normpdf(xx, est_mu[1], est_vars[1])


# GMM
iter_num = 50
for i_iter in range(iter_num):
    """
    E-step
    - 이 상황에서는 파이, 평균, 분산은 고정되 있다.
    - 각 데이터가 어느 분포에 속할 것인지 확률을 구하자
    """
    for i_data in range(n_data):
        l0 = normpdf(X[i_data], est_mu[0], est_vars[0])
        l1 = normpdf(X[i_data], est_mu[1], est_vars[1])

        est_w[i_data][0] = (l0 * est_phi[0]) / (l0 * est_phi[0] + l1 * est_phi[1])
        est_w[i_data][1] = (l1 * est_phi[1]) / (l0 * est_phi[0] + l1 * est_phi[1])


    """
    M-step
    - 데이터가 어디에 속할 건지 확률 적으로 알고있다.

    """
    est_w_sum = sum(est_w, 0) 

    # 파이 추정
    est_phi = 1/n_data * est_w_sum

    # 평균 추정
    est_mu = np.dot(X, est_w) / est_w_sum
    """
    - 아래 연산을 한번에 수행 하도록 했다.
    est_mu[0] = np.dot(X,est_w[:,0]) / (est_w_sum[0])
    est_mu[1] = np.dot(X,est_w[:,1]) / (est_w_sum[1])
    """

    # 분산 추정
    est_vars[0] = np.sum(est_w[:, 0] * (X - est_mu[0])**2) / np.sum(est_w[:, 0])
    est_vars[1] = np.sum(est_w[:, 1] * (X - est_mu[1])**2) / np.sum(est_w[:, 1])

    est_yy1 = normpdf(xx, est_mu[0], est_vars[0])
    est_yy2 = normpdf(xx, est_mu[1], est_vars[1])

    plt.plot(xx, est_yy1 * est_phi[0], 'r', linewidth=2)
    plt.plot(xx, est_yy2 * est_phi[1], 'g', linewidth=2)
    plt.hist(X, bins=50, density=True, color='skyblue', edgecolor='black')

    plt.xlabel('x')
    plt.ylabel('pdf')
    plt.grid(True)

    plt.savefig(f".\\result\\{i_iter}.png")
    plt.clf()

