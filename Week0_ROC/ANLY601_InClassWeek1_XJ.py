#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:46:29 2019

@author: xungejiang
"""

# -*- coding: utf-8 -*-
"""
In Class Exercise (submit before end of class for +10 extra credit).
******************************************************
Goal: Implement function 
ROC, AUC = createROC(predictedClass, actualClass)
******************************************************

which computes a ROC given list of predicted classes and list of actualClasses.
Specifically construct ROC as a dictionary with entries for "tpr" and "fpr": the 
arrays of the true positive rates and false positive rates.

This function will also compute the area under the curve, AUC using simple Riemann 
estimate.

Compare your results to the existing methods roc_curve and auc in sklearn, by overlaying
the ROCs computed by your function (in blue) and the ROCs computed by sklearn (in black). 
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
 

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Binarize the output
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# In Class Assignment 1
prob = classifier.predict_proba(X_test)
def createROC(predictedClass, actualClass):
    TP_list = []
    FP_list = []
    ROC2 = dict()
    TP = 0
    FP = 0       
    # build threshold
    threshold = np.linspace(0, predictedClass.max(), num = 75)
    # count total TP and FP
    for seq in threshold:
        TP = sum(np.logical_and(predictedClass > seq, actualClass == 1))
        FP = sum(np.logical_and(predictedClass > seq, actualClass == 0))
        # save TP count into a list
        TP_list.append(TP)
        TP = 0 
        # save FP count into a list
        FP_list.append(FP)
        FP = 0
    # calculate tpr and fpr
    tpr2 = TP_list/sum(actualClass==1)
    fpr2 = FP_list/sum(actualClass==0)
    # create a dictionary - ROC
    ROC2['tpr'] = tpr2
    ROC2['fpr'] = fpr2
    AUC2 = 0
    # compute auc by sum(tpr * diff(fpr))
    for i in range(74):
        AUC2 += (tpr2[i])*(fpr2[i]-fpr2[i+1])
        
    return(ROC2, AUC2)

ROC2 = dict()
roc_auc2 = dict()
#for i in range(n_classes):
#    ROC2[i], roc_auc2[i] = createROC(prob[:, i], y_test[:, i])
ROC2[2], roc_auc2[2] = createROC(prob[:, 2], y_test[:, 2])

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='black',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])

# Display your results here.
# compare to roc_curve function
plt.plot(ROC2[2]['fpr'], ROC2[2]['tpr'], color='blue',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc2[2])

plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()