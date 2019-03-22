#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:06:40 2019

@author: xungejiang
"""

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix   
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import label_binarize
from scipy import stats
import math
################################### Part a #################################
iris = datasets.load_iris()

#create a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = pd.DataFrame(iris.target)

# extract class label 1 = versicolor, 2 = virginica
df = df[50:150]
df.columns = ['feature1', 'feature2', 'feature3', 'feature4', 'class']

# KL 
model34 = np.array([0.52, 0.48])
model13 = np.array([0.5, 0.5])
kl12 = stats.entropy(df['feature1'], df['feature2'])
print("KL for feature pair 1 and 2:", kl12)

kl34 = stats.entropy(df['feature3'], df['feature4'])
print("KL for feature pair 3 and 4:", kl34)

kl13 = stats.entropy(df['feature1'], df['feature3'])
print("KL for feature pair 1 and 3:", kl13)

# FDR 
def FDR_calc(class1, class2): 

    mean1=np.mean(class1) # class 1
    mean2=np.mean(class2) # class 2

    var1=np.var(class1) # class 1
    var2=np.var(class2) # class 2
    
    mean_diff = np.dot((mean1-mean2),(mean1-mean2))
    var_diff = np.dot((var1+var2), (var1+var2))

    fdr = mean_diff / var_diff
    
    return fdr

class1 = df[1:50].loc[:, ['feature1','feature2']]
class2 = df[50:100].loc[:, ['feature1','feature2']]
fdr12 = FDR_calc(class1, class2)
print("FDR for feature pair 1 and 2:", fdr12)

class1 = df[1:50].loc[:, ['feature3','feature4']]
class2 = df[50:100].loc[:, ['feature3','feature4']]
fdr34 = FDR_calc(class1, class2)
print("FDR for feature pair 3 and 4:", fdr34)

class1 = df[1:50].loc[:, ['feature1','feature3']]
class2 = df[50:100].loc[:, ['feature1','feature3']]
fdr13 = FDR_calc(class1, class2)
print("FDR for feature pair 1 and 3:", fdr13)

# Bhattacharyya Distance (B)
def bhatta(hist1, hist2):
    # calculate mean of hist1
    h1_ = np.mean(hist1)
    # calculate mean of hist2
    h2_ = np.mean(hist2)

    # calculate score
    score = 0;
    for i in range(len(hist1)):
        score += math.sqrt(hist1[i] * hist2[i]);
    # print h1_,h2_,score;
    score = math.sqrt( 1 - ( 1 / math.sqrt(h1_*h2_*len(hist1)*len(hist1)) ) * score);
    return score;

b12 = bhatta(np.array(df['feature1']), np.array(df['feature2']))
print("Bhattacharyya Distance for feature pair 1 and 2 is:", b12)

b34 = bhatta(np.array(df['feature3']), np.array(df['feature4']))
print("Bhattacharyya Distance for feature pair 3 and 4 is:", b34)

b13 = bhatta(np.array(df['feature1']), np.array(df['feature3']))
print("Bhattacharyya Distance for feature pair 1 and 3 is:", b13)


plt.plot([kl12,kl34,kl13], 'o',color='red')
plt.plot([fdr12,fdr34,fdr13], 'o',color='blue')
plt.plot([b12,b34,b13], 'o', color='black')
plt.title('Access Values')

################################### Part b #################################

from sklearn.metrics import roc_curve, auc
def plotROC(X, y, name):
    clf = LogisticRegressionCV(cv=5, multi_class='ovr').fit(X, y)
    y = label_binarize(y, classes=[0, 1, 2])[:, 1:3]
    y_pred = clf.predict(X)
    y_pred = label_binarize(y_pred, classes=[0, 1, 2])[:, 1:3]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])  
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='blue',
         lw=lw, label='ROC Curve Class 1 (area = %0.2f)' % roc_auc[0])
    plt.plot(fpr[1], tpr[1], color='black',
         lw=lw, label='ROC Curve Class 2 (area = %0.2f)' % roc_auc[1])

    plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve For '+str(name))
    plt.legend(loc="lower right")
    plt.show()

# 1) Use feature pair 1 and 2
X = df.loc[:, ['feature1','feature2']]
y = df.iloc[:, 4]
plotROC(X, y, 'Feature 1 and 2')

# 2) Use feature pair 3 and 4
X = df.loc[:, ['feature3','feature4']]
y = df.iloc[:, 4]
plotROC(X, y, 'Feature 3 and 4')

# 3) Use feature pair 1 and 3
X = df.loc[:, ['feature1','feature3']]
y = df.iloc[:, 4]
plotROC(X, y, 'Feature 1 and 3')

################################### Part c #################################    

X = df.loc[:, ['feature1','feature2','feature3','feature4']]
y = df.loc[:, ['class']]
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True, n_components = 1)
y_pred = lda.fit(X, y).predict(X)

y = label_binarize(y, classes=[0, 1, 2])[:, 1:3]
y_pred = label_binarize(y_pred, classes=[0, 1, 2])[:, 1:3]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])  
    
plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='blue',
    lw=lw, label='ROC Curve Class 1 (area = %0.2f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], color='black',
    lw=lw, label='ROC Curve Class 2 (area = %0.2f)' % roc_auc[1])

plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve LDA')
plt.legend(loc="lower right")
plt.show()


att12=pd.concat([df['feature1'],df['feature2'],df['class']],axis=1)
c1=att12[att12['class']==1]
c2=att12[att12['class']==2]
z1=(c1.iloc[:,0:2]).mean()
z2=(c2.iloc[:,0:2]).mean()
diff1=(c1.iloc[:,0:2]-z1).reset_index(drop=True)
s1=0
for i in range(len(diff1)):
    s1=s1+np.dot(diff1.loc[i,],diff1.loc[i,])
    s1=s1/len(diff1)