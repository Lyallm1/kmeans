from sklearn.cluster import KMeans
from scipy import random

r = random.random()
X = [[1.,  0.75,  1.125], [1.,  1.75,  1.125], [-1., -1.25, -0.875], [-1., -1.25, -1.375]]
print(X)
kmeans = KMeans(2, random_state=10, max_iter=1).fit(X)
print(kmeans.cluster_centers_)
