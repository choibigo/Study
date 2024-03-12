import os
from shutil import copyfile
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 그래픽스를 위한 모듈
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

K = 10
save_root_path = os.path.join(r"C:\Users\cbigo\Desktop\temp", f"k_{K}")
os.makedirs(save_root_path, exist_ok=True)

for label in range(K):
    os.makedirs(os.path.join(save_root_path, f"{label}"), exist_ok=True)

interval = 1
data = np.load('feature_2_27_1.npy')[::interval]

# K-means 클러스터링

kmeans = KMeans(n_clusters=K, random_state=42)
cluster_labels = kmeans.fit_predict(data)

# t-SNE를 사용하여 3차원으로 축소
tsne = TSNE(n_components=3, random_state=42)
tsne_result = tsne.fit_transform(data)

# 시각화
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# K-means 클러스터의 색상 설정
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'slategray', 'lightpink', 'indigo', 'khaki', 'cyan', 'coral', 'lightyellow', 'tan', 'slateblue']
cluster_colors = [colors[label] for label in cluster_labels]

image_root_path = r"D:\workspace\Difficult\dataset\behavior_scene_image\rn_int_leaf_pencase"
for image_num, label in enumerate(cluster_labels):
    image_num *= interval
    copyfile(os.path.join(image_root_path, f"{image_num}.png", ), os.path.join(save_root_path, f"{label}", f"{image_num}.png"))

# t-SNE 결과를 3D scatter plot으로 표시
ax.scatter(tsne_result[:, 0], tsne_result[:, 1], tsne_result[:, 2], c=cluster_colors, alpha=0.7)

ax.set_title('t-SNE Visualization of K-means Clustering in 3D')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_zlabel('t-SNE Dimension 3')

plt.show()