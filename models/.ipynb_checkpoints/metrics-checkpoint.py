import numpy as np

from math import sqrt       
    
def mMAE(real, pred):
    res = np.mean(abs(real - pred))
    return res

def mMSE(real, pred):
    res = np.mean((real-pred)**2)
    return res
    

def mRMSE(real, pred):
    res = sqrt(mMSE(real, pred))
    return res

def mMAPE(real,pred):
    res = np.mean((abs(real-pred))/real)
    return res

def mR2(real, pred): 
    res = 1 - ((mMSE(real, pred))/(np.mean((real - np.mean(real))**2)))
    return res
    
