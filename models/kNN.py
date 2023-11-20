from math import sqrt
from statistics import mode
import numpy as np

class kNN:
    def __init__(self, k):
        self.k = k


    def fit(self, x_train, y_train):
        self.X = x_train
        self.Y = y_train


    def predict(self, x_test):
        pred = []
        for i in range(0, len(x_test)):
            distances = []
            for j in range(0, len(self.X)):
                distances.append([self.dist(x_test[i], self.X[j]), self.Y[j]])
            distances.sort(key=lambda x: x[0])
            short_dist = distances[:self.k]
            list_of_y = [l[1] for l in short_dist]                      
            pred.append(mode(list_of_y))
        return pred


    def dist(self, x1, x2):
        dis = 0 
        for i in range(0, len(x1)):
            dis += (x1[i]-x2[i])**2
        dis = sqrt(dis)  
        return dis
