import numpy as np
class myPCA:
    def __init__(self, n_comp):
        self.n_comp = n_comp
    def fit_transform(self, X):
        
        X = ((X - X.mean()) / X.std())
        
        X_cov = np.cov(X.T)
        
        eigenValues, eigenVectors = np.linalg.eig(X_cov)
        
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        
        eigenValues = eigenValues[:self.n_comp]
        eigenVectors = eigenVectors[:self.n_comp]

        matrix = np.column_stack(eigenVectors)
        
        result = np.dot(X, matrix)
        return result
