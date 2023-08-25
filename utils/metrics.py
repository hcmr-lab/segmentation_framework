

from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
import numpy as np


class multi_class_metrics():
    def __init__(self,classes):
        self.classes = classes
        self.iou = []
        self.f1 = []


    def compute_multi_class_metrics(self,true_labels,pred):
        iou = jaccard_score(true_labels.flatten(), pred.flatten(),average= "micro",labels=self.classes,zero_division=0) #  average='macro'
        f1  = f1_score(true_labels.flatten(), pred.flatten(),average= "micro",labels=self.classes, zero_division=0)
        self.iou.append(iou)
        self.f1.append(f1)

        return {'iou':iou,'f1':f1}
    
    def get_global_results(self):
        return {'iou':np.mean(self.iou),'f1':np.mean(self.f1)}