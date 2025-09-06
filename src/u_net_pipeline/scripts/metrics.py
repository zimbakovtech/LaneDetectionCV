import numpy as np

def dice(y_true, y_pred, smooth=1e-6):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    inter = np.sum(y_true * y_pred)
    return (2*inter + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def iou(y_true, y_pred, smooth=1e-6):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    inter = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - inter
    return (inter + smooth) / (union + smooth)
