import numpy as np
from scipy.stats import mode

class CART:
    def __init__(self, criterion, max_depth, min_samples_split):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        
    def fit(self, X, y):
        self.tree = self.build_tree(X, y, depth = 0)

    def build_tree(self, X, y, depth):
        if(depth>=self.max_depth) or (len(X)<= self.min_samples_split):
            if(self.criterion == 'mse'):
                leaf_value = np.mean(y)
            if(self.criterion == 'gini'):
                leaf_value = mode(y)[0]
            return {
                'leaf': 1,
                'value': leaf_value
                }
                
        best_column, best_threshold = self.find_best_split(X, y)

        left_child = X[:, best_column] <= best_threshold
        right_child = X[:, best_column] > best_threshold

        node = {
            'column' : best_column,
            'threshold' : best_threshold,
            'left' : self.build_tree(X[left_child], y[left_child], depth + 1),
            'right' : self.build_tree(X[right_child], y[right_child], depth + 1)
        }
        return node

    def gini(self, left_y, right_y):
        left_gini = 1 - np.sum((np.bincount(left_y) / len(left_y))**2)
        right_gini = 1 - np.sum((np.bincount(right_y) / len(right_y))**2)
        gini = (len(left_y) * left_gini + len(right_y) * right_gini) / (len(left_y) + len(right_y))
        return gini

    def mse(self, left_y, right_y):
        left_mse = np.mean((left_y - np.mean(left_y))**2)
        right_mse = np.mean((right_y - np.mean(right_y))**2)
        mse = (len(left_y) * left_mse + len(right_y) * right_mse) / (len(left_y) + len(right_y))
        return mse
    
    def find_best_split(self, X, y):
        best_metric = 9999999
        best_column = None
        best_threshold = None
        
        for column in range(X.shape[1]):
            thresholds = np.unique(X[:, column])
            
            for threshold in thresholds:
                left = X[:, column] <= threshold
                right = X[:, column] > threshold

                if(self.criterion == 'mse'):
                    loss_func_value = self.mse(y[left], y[right])
                if(self.criterion == 'gini'):
                    loss_func_value = self.gini(y[left], y[right])

                if loss_func_value < best_metric:
                    best_metric = loss_func_value
                    best_column = column
                    best_threshold = threshold
        
        return best_column, best_threshold
    
    def predict(self, x_test):
        return np.array([self.tree_search(x, self.tree) for x in x_test])
    
        
    def tree_search(self, x, node):  
        if 'leaf' in node:
            return node['value']      
        if x[node['column']] <= node['threshold']:
            return self.tree_search(x, node['left'])
        else:
            return self.tree_search(x, node['right'])
