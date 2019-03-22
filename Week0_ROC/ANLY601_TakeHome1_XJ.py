#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:46:29 2019

@author: xungejiang
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 18:10:29 2018

Take-Home Assignment (Extra Credit +20).
******************************************************
Read: ROC Analysis paper posted on course site.
To Do: Implement two more ROC curve creating functions with error bars
******************************************************

Specifically, the creation of a ROC curve can be seen as a trial, the results of which
may vary based on the variability of data. The ROC analysis paper acknowledges this 
concern and addresses it by designing two algorithms which result in ROCs with error bars:
one algorithm addressing variability in TPR and one addressing variability in FPR.

Goals: Implement functions
ROC, AUC = createROCtpr(predictedClass, actualClass)
ROC, AUC = createROCfpr(predictedClass, actualClass)

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


prob = classifier.predict_proba(X_test)
def createROC(predictedClass, actualClass):
   
    TP_list = []
    FP_list = []
    TP = 0
    FP = 0       
    threshold = np.linspace(0, predictedClass.max(), num = 75)
    for seq in threshold:
        TP = sum(np.logical_and(predictedClass > seq, actualClass == 1))
        FP = sum(np.logical_and(predictedClass > seq, actualClass == 0))
        TP_list.append(TP)
        TP = 0 
        FP_list.append(FP)
        FP = 0
    tpr2 = TP_list/sum(actualClass==1)
    fpr2 = FP_list/sum(actualClass==0)
    auc2 = 0
    for i in range(74):
        auc2 += (tpr2[i])*(fpr2[i]-fpr2[i+1])
    return(fpr2, tpr2, auc2)

fpr2 = dict()
tpr2 = dict()
roc_auc2 = dict()
for i in range(n_classes):
    fpr2[i], tpr2[i], roc_auc2[i] = createROC(prob[:, i], y_test[:, i])
fpr2[2],tpr2[2], roc_auc2[2] = createROC(prob[:, 2], y_test[:, 2])


X_train = dict()
X_test = dict()
y_train = dict()
y_test = dict()


fpr3 = dict()
tpr3 = dict()
def createROCfpr(predictedClass, actualClass):
   
    TP_list = []
    FP_list = []
    TP = 0
    FP = 0       
    threshold = np.linspace(0, predictedClass.max(), num = 75)
    for seq in threshold:
        TP = sum(np.logical_and(predictedClass > seq, actualClass == 1))
        FP = sum(np.logical_and(predictedClass > seq, actualClass == 0))
        TP_list.append(TP)
        TP = 0 
        FP_list.append(FP)
        FP = 0
    tpr3 = TP_list/sum(actualClass==1)
    fpr3 = FP_list/sum(actualClass==0)
    return(fpr3, tpr3)
    
# generate 5 samples 
n_samples = 5
for i in range(n_samples):
    X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(X, y, test_size=.5, random_state=None)
   
prob3 = dict()
for i in range(n_samples):
    prob3[i] = classifier.predict_proba(X_test[i])

# specific for class 2
fpr3 = [] 
tpr3 = []
for j in range(n_samples):
    resultfpr, resulttpr = createROCfpr(prob3[j][:, 2], y_test[j][:, 2])
    fpr3.append(resultfpr)
    tpr3.append(resulttpr)
    # find average of fpr
    mean_fpr = np.mean(fpr3[0:n_samples][:], axis = 0)
    mean_tpr = np.mean(tpr3[0:n_samples][:], axis = 0)
    
def interpolate(tpr1, tpr2, fpr1, fpr2, x):
    slope = (tpr2-tpr1) / (fpr2 - fpr1)
    result = tpr1 + slope*(x - fpr1)
    return result
for i in range(75):
    if fpr3[j][i] == mean_fpr[i]: continue
    if fpr3[j][i] == mean_fpr[i]:
        y = interpolate()
            
mean_tpr[0] = 1.0
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)
std_tpr = np.std(tpr3, axis = 0)
plt.plot(mean_fpr, mean_tpr, color='b')

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
