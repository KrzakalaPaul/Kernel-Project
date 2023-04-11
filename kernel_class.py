import numpy as np
from sklearn.svm import SVC
from SVM import SVM
import pickle as pkl

with open('training_labels.pkl', 'rb') as file:
    train_labels = pkl.load(file)

class Kernel():

    def __init__(self):
        self.kernel_train_train=None
        self.kernel_validation_train=None
    
    def fit(self,train_indices=None,C=1,class_weight='balanced'):

        #self.clf = SVC(kernel='precomputed',C=C, probability=True)
        self.clf = SVM(C=C,class_weight=class_weight)
        self.train_indices=train_indices

        if train_indices is None:
            kernel_train=self.kernel_train_train
            y_train=train_labels
        else:
            kernel_train=self.kernel_train_train[train_indices][:,train_indices]
            y_train=train_labels[train_indices]

        self.clf.fit( kernel_train, y_train )

    def pred_train(self):

        if self.train_indices is None:
            return self.clf.predict_log_proba(self.kernel_train_train)
        else:
            kernel_test_train=self.kernel_train_train[self.train_indices][:,self.train_indices]
            return self.clf.predict_log_proba(kernel_test_train)
            

    def pred(self,test_indices=None):
        
        if test_indices is None:
            return self.clf.predict_log_proba(self.kernel_validation_train)
        else:
            kernel_test_train=self.kernel_train_train[test_indices][:,self.train_indices]
            return self.clf.predict_log_proba(kernel_test_train)



