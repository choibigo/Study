from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

data = load_digits()

print(data['data'])
print(data['data'].shape)

n_components = 2
model = TSNE(n_components=n_components)
X_embedded = model.fit_transform(data.data)



plt.scatter(X_embedded[:,0], X_embedded[:,1])
plt.show()