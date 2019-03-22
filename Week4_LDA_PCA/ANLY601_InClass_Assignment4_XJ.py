#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 12:05:59 2019

@author: xungejiang
"""

"""

2) Update the script so that 3 different data sets are created. Specifically,
    a) Construct a synthetic data set where both the LDA and PCA results are 
        similar and best case.
    b) Construct a synthetic data set where both the LDA and PCA results are 
        similar and worst case.
    c) Construct a synthetic data set where the LDA results are best case,
        but the PCA results are worst case.
********************************************************


@author: jerem
"""



from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn import linear_model
# If your interested ... 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

# ############################## PART 1 #######################################
# Colormap
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)


# #############################################################################
# Generate datasets for illustrative purposes
def dataset_fixed_cov():
    '''Generate 2 Gaussians samples with the same covariance matrix'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -0.23], [0.83, .23]])
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


def dataset_cov():
    '''Generate 2 Gaussians samples with different covariance matrices'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -1.], [2.5, .7]]) * 2.
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


# #############################################################################
# Plot functions
def plot_data(lda, X, y, y_pred, fig_index):
    splot = plt.subplot(2, 2, fig_index)
    if fig_index == 1:
        plt.title('Linear Discriminant Analysis')
        plt.ylabel('Data with\n fixed covariance')
    elif fig_index == 2:
        plt.title('Quadratic Discriminant Analysis')
    elif fig_index == 3:
        plt.ylabel('Data with\n diff covariances')

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    alpha = 0.5

    # class 0: dots
    plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', alpha=alpha,
             color='red', markeredgecolor='k')
    plt.plot(X0_fp[:, 0], X0_fp[:, 1], '*', alpha=alpha,
             color='#990000', markeredgecolor='k')  # dark red

    # class 1: dots
    plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', alpha=alpha,
             color='blue', markeredgecolor='k')
    plt.plot(X1_fp[:, 0], X1_fp[:, 1], '*', alpha=alpha,
             color='#000099', markeredgecolor='k')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.))
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

    # means
    plt.plot(lda.means_[0][0], lda.means_[0][1],
             'o', color='black', markersize=10, markeredgecolor='k')
    plt.plot(lda.means_[1][0], lda.means_[1][1],
             'o', color='black', markersize=10, markeredgecolor='k')

    return splot

def plot_data_pca(lda, X, y, y_pred, fig_index): 
    splot = plt.subplot(2, 2, fig_index)
    
    if fig_index == 1:
        plt.title('Linear Discriminant Analysis')
        plt.ylabel('Data with\n fixed covariance') 
    elif fig_index == 2:
        plt.title('PCA + Linear Classifier') 
    elif fig_index == 3:
        plt.ylabel('Data with\n diff covariances')
    tp = (y == y_pred) # True Positive 
    tp0, tp1 = tp[y == 0], tp[y == 1] 
    X0, X1 = X[y == 0], X[y == 1] 
    X0_tp, X0_fp = X0[tp0], X0[~tp0] 
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
    
    alpha = 0.5
    # class 0: dots
    plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', alpha=alpha,
             color='red', markeredgecolor='k')
    plt.plot(X0_fp[:, 0], X0_fp[:, 1], '*', alpha=alpha,
             color='#990000', markeredgecolor='k') # dark red
    # class 1: dots
    plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', alpha=alpha,
             color='blue', markeredgecolor='k')
    plt.plot(X1_fp[:, 0], X1_fp[:, 1], '*', alpha=alpha,
             color='#000099', markeredgecolor='k') # dark blue
    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    pca = PCA(n_components=1)
    Z = lda.predict_proba(pca.fit(X).transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.))
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

    return splot

def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                              180 + angle, facecolor=color,
                              edgecolor='yellow',
                              linewidth=2, zorder=2)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
    splot.set_xticks(())
    splot.set_yticks(())


def plot_lda_cov(lda, splot):
    plot_ellipse(splot, lda.means_[0], lda.covariance_, 'red')
    plot_ellipse(splot, lda.means_[1], lda.covariance_, 'blue')


def plot_qda_cov(qda, splot):
    plot_ellipse(splot, qda.means_[0], qda.covariance_[0], 'red')
    plot_ellipse(splot, qda.means_[1], qda.covariance_[1], 'blue')


# Perform LDA

for i, (X, y) in enumerate([dataset_fixed_cov(), dataset_cov()]):
    # Linear Discriminant Analysis
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
    y_pred = lda.fit(X, y).predict(X)
    splot = plot_data(lda, X, y, y_pred, fig_index=2 * i + 1)
    plot_lda_cov(lda, splot)
    plt.axis('tight')

    # Run PCA, Train Linear Classifier - Use Logisitic Regression
    regr = linear_model.LogisticRegression()
    pca = PCA(n_components=1)
    X_r = pca.fit(X).transform(X)
    regr.fit(X_r, y)
    y_pred = regr.predict(X_r)
    ...
    ...
    splot = plot_data_pca(regr, X, y, y_pred, fig_index=2 * i + 2)
     
    # Quadratic Discriminant Analysis
    # qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    # y_pred = qda.fit(X, y).predict(X)
    
  #  plot_qda_cov(qda, splot)
    plt.axis('tight')
    
plt.suptitle('Linear Discriminant Analysis vs PCA + simple classifier'
             'Analysis')
plt.show()

# ############################## PART 2 #######################################   
# a) both the LDA and PCA results are similar and best case.
X, y = dataset([1, 1], [2, 2], [[0, 0.1],[0.1, 0]])
# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
y_pred = lda.fit(X, y).predict(X)
splot = plot_data(lda, X, y, y_pred,fig_index = 1)
plot_lda_cov(lda, splot)
plt.axis('tight')

# PCA and Linear Classifier
regr = linear_model.LogisticRegression()
pca = PCA(n_components=1)
X_r = pca.fit(X).transform(X)
regr.fit(X_r, y) 
y_pred = regr.predict(X_r)
splot = plot_data_pca(regr, X, y, y_pred, fig_index = 2)
plt.axis('tight')
plt.suptitle('LDA vs PCA + Logistic Classifier, Both Good')
plt.show()

# We can see the boundary and classifier is very clear. 

# b) both the LDA and PCA results are similar and worst case.
X, y = dataset([50, 50], [-50, -50], [[200, 500],[500, 200]])
# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
y_pred = lda.fit(X, y).predict(X)
splot = plot_data(lda, X, y, y_pred,fig_index = 1)
plot_lda_cov(lda, splot)
plt.axis('tight')

# PCA and Linear Classifier
regr = linear_model.LogisticRegression()
pca = PCA(n_components=1)
X_r = pca.fit(X).transform(X)
regr.fit(X_r, y) 
y_pred = regr.predict(X_r)
splot = plot_data_pca(regr, X, y, y_pred, fig_index = 2)
plt.axis('tight')
plt.suptitle('LDA vs PCA + Logistic Classifier')
plt.show()

# We can see there is a lot of FP and FNs. 

# c) LDA results are best case,but the PCA results are worst case.
X, y = dataset([0.5, 0.5], [5, 5], [[1, 0.5],[0.5, 5]])
# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
y_pred = lda.fit(X, y).predict(X)
splot = plot_data(lda, X, y, y_pred,fig_index = 1)
plot_lda_cov(lda, splot)
plt.axis('tight')

# PCA and Linear Classifier
regr = linear_model.LogisticRegression()
pca = PCA(n_components=1)
X_r = pca.fit(X).transform(X)
regr.fit(X_r, y) 
y_pred = regr.predict(X_r)
splot = plot_data_pca(regr, X, y, y_pred, fig_index = 2)
plt.axis('tight')
plt.suptitle('LDA vs PCA + Logistic Classifier')
plt.show()

# We can see LDA is performing good while PCA doesn't seem to separate good. 