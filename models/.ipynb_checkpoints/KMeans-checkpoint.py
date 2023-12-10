from math import sqrt
import numpy as np

class myKMeans:
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        
    def fit(self, X):
        self.m, self.n = np.shape(X)
        
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for _ in range(self.max_iter):
            clusters_idx = np.empty(self.m)
            
            for i in range(0, self.m):
                clusters_idx[i] = np.argmin(self.dist(X[i], self.centroids))

            new_centroids = np.empty(self.k)

            for i in range(0, self.k):
                p = X[clusters_idx == i]
                new_centroids[i] = np.mean(p)
                
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, X):
	m, n = np.shape(X)
	for i in range (0, m)
            dis[i] = np.argmin(self.dist(X[i], self.centroids))
        return dis

    def dist(self, x, y):
        dis = np.empty(self.k)
        for i in range(0, self.k):
            dis[i] = sqrt(np.sum((y[i] - x) ** 2))  
        return dis
