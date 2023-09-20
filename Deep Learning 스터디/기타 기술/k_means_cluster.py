import random
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

K = 3

if __name__ == "__main__":

    np.random.seed(13)
    random.seed(13)

    x, y = make_blobs(n_samples=50, centers=K, cluster_std=3)

    k_means = KMeans(init='random', n_clusters=K, n_init=12)
    k_means.fit(x)

    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_

    fig = plt.figure(figsize=(6, 4))

    # 레이블 수에 따라 색상 배열 생성
    colors = np.array([[1 if x == 3 else random.random() for x in range(4) ] for _ in range(len(set(k_means_labels)))])
    ax = fig.add_subplot(1, 1, 1)

    for k, col in zip(range(4), colors):
        my_members = (k_means_labels == k)

        cluster_center = k_means_cluster_centers[k]

        ax.plot(x[my_members, 0], x[my_members, 1], 'w', markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

    ax.set_title('K-Means')
    ax.set_xticks(())
    ax.set_yticks(())

    plt.scatter(x[:, 0], x[:, 1], marker='.')
    plt.show()

