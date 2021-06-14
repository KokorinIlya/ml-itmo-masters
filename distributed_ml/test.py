from sklearn.cluster import KMeans
import torch

clust = KMeans(n_clusters=3, init='k-means++')
n, k = 10, 5
X = torch.randn(size=(n, k))
clust.fit(X)
print(clust.labels_)
centers = torch.from_numpy(clust.cluster_centers_).float()
print(centers)
X_t = torch.zeros_like(X)
X_t[list(range(n))] = centers[clust.labels_]
print(X_t)
