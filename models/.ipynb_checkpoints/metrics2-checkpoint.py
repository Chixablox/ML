import numpy as np

def mConfusionMatrix(y_test, y_pred):
    cm = np.matrix('0 0; 0 0')
    for i in range (0, len(y_pred)):
        if ((y_test[i] == 0) and (y_pred[i] == 0)):
            cm[0,0] += 1
        elif ((y_test[i] == 0) and (y_pred[i] == 1)):
            cm[0,1] += 1
        elif ((y_test[i] == 1) and (y_pred[i] == 0)):
            cm[1,0] += 1
        elif ((y_test[i] == 1) and (y_pred[i] == 1)):
            cm[1,1] += 1
    return cm

   
def mAccuracy(y_test, y_pred):
     cm = mConfusionMatrix(y_test, y_pred)
     accuracy = (cm[0,0]+cm[1,1])/len(y_test)
     return accuracy


def mPrecision(y_test, y_pred):
    cm = mConfusionMatrix(y_test, y_pred)
    precision_0 = cm[0,0]/(cm[0,0]+cm[1,0])
    precision_1 = cm[1,1]/(cm[1,1]+cm[0,1])
    return precision_0, precision_1


def mRecall(y_test, y_pred):
    cm = mConfusionMatrix(y_test, y_pred)
    recall_0 = cm[0,0]/(cm[0,0]+cm[0,1])
    recall_1 = cm[1,1]/(cm[1,1]+cm[1,0])
    return recall_0, recall_1


def mF1Score(y_test, y_pred):
    precision_0, precision_1 = mPrecision(y_test, y_pred)
    recall_0, recall_1 = mRecall(y_test, y_pred)
    
    f1_0 = (2*recall_0*precision_0)/(recall_0+precision_0)
    f1_1 = (2*recall_1*precision_1)/(recall_1+precision_1)
    return f1_0, f1_1
    

