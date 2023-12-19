import numpy as np
class myPCA:
    def __init__(self, n_comp):
        self.n_comp = n_comp
    def transform(self, X):
        #стандартизация данных
        X = ((X - X.mean()) / X.std())
        #вычисление ковариационной матрицы
        X_cov = np.cov(X.T)
        #вычисление собственных значений и векторов матрицы
        eigenValues, eigenVectors = np.linalg.eig(X_cov)
        #сортировка пар <собственное значение; собственный вектор по убыванию>
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        #выбор первых n_comp пар
        eigenValues = eigenValues[:self.n_comp]
        eigenVectors = eigenVectors[:self.n_comp]
        #составление матрицы из векторов-столбцов
        matrix = np.column_stack(eigenVectors)
        #умножение стандартизированной матрицы на матрицу из прошлого пункта
        result = np.dot(X, matrix)
        return result
