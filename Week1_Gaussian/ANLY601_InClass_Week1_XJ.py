#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 12:27:51 2019

@author: xungejiang
"""

# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 18:10:29 2018


In Class Exercise:  xx/100  (+ 10 extra credit if submitted before end of day of class)
******************************************************
Assume: Single Gaussian
Goal: Implement functions, given data X.
mu0, sigma0 = gaussianML(X)
mu1, sigmaGuess = gaussianMAP(X, priorMu, sigmaGuess)
******************************************************

The functions should compute the ML and MAP estimates respectively. Plot
the results of the learned parameters (using a contour like display) for the 
purposes of comparison. Feel free  to use the code snippets provided below
or feel free to edit as you see fit. Assume covariance matrices are diagonal. 
Also assume \mu has Gaussian Prior. We will not solve for the MAP covariance,
instead, We will discuss more when we dive into Bayesian 
Methods. 

Construct synthetic 2-d data X or find a simple data set to use to test your code.
********************************************************
"""

import pandas as pd


import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

# Training Data
# Choose a training data set.

# generate train set, store in X
np.random.seed(seed=12)
mu, sigma = 0, 1
X = pd.DataFrame()
X['0'] = np.random.normal(mu, sigma, 50)
X['1'] = np.random.normal(mu, sigma, 50)

# ML estimate of 2d Gaussian
def gaussianML(X):
    X = pd.DataFrame(X)
    # mu set and sigma set
    mu0 = sum(X['0'])/len(X)
    sigma0 = (sum((X['0']-mu0)**2.0) / len(X))**0.5
    mu1 = sum(X['1'])/len(X)
    sigma1 = (sum((X['1']-mu1)**2.0) / len(X))**0.5
    mu = [mu0, mu1]
    sigma = [sigma0, sigma1]
    
    return(mu, sigma)
    
def gaussianMAP(X, priorMu, sigmaGuess):
    X = pd.DataFrame(X)
    
    # use the same sigma set as ML to derive mu set
    sigma0 = (sum((X['0']-sum(X['0'])/len(X))**2.0) / len(X))**0.5
    sigma1 = (sum((X['1']-sum(X['1'])/len(X))**2.0) / len(X))**0.5
    mu0 = (priorMu[0] + (sigmaGuess[0]**2 / sigma0**2) * sum(X['0'])) / (1 + (sigmaGuess[0]**2 / sigma0**2)* len(X))
    mu1 = (priorMu[1] + (sigmaGuess[1]**2 / sigma1**2) * sum(X['1'])) / (1 + (sigmaGuess[1]**2 / sigma1**2)* len(X))

    # calculate sigma set
    sigma0 = (sum((X['0']-mu0)**2.0) / len(X))**0.5
    sigma1 = (sum((X['1']-mu1)**2.0) / len(X))**0.5

    mu = [mu0, mu1]
    sigma = [sigma0, sigma1]
    return(mu, sigma)  

def gaussian_2d(x, y, x0, y0, xsig, ysig):
    return np.exp(-0.5*(((x-x0) / xsig)**2 + ((y-y0) / ysig)**2))

#print(gaussian_2d(X))
def plt_2dGaussians(mu0, mu1, sig0, sig1):
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = gaussian_2d(X, Y, mu0[0], mu0[1], sig0[0],sig0[1])
    Z2 = gaussian_2d(X, Y, mu1[0], mu1[1],  sig1[0],sig1[1])

    # Create a contour plot with labels using default colors.  The
    # inline argument to clabel will control whether the labels are draw
    # over the line segments of the contour, removing the lines beneath
    # the label
    
    # For use help, see https://matplotlib.org/
    plt.clf()
    plt.figure(figsize=(15,8))
    CS1 = plt.contour(X, Y, Z2)
    plt.clabel(CS1, inline=1, fontsize=10)
    CS2 = plt.contour(X, Y, Z1)
    plt.clabel(CS2, inline=1, fontsize=10)
    
    plt.title('Learned Gaussian Contours')

# Main
    
# generate priormu and sigmaguess for MAP method
prior = pd.DataFrame()
prior['0'] = np.random.normal(1, 1, 2)
prior['1'] = np.random.normal(1, 1, 2)
priormu = np.mean(prior, axis = 0)
sigmaguess = np.std(prior, axis = 0)

# extract 2 mus and 2 sigmas to plot
mu0 = gaussianML(X)[0]
mu1 = gaussianMAP(X,priormu ,sigmaguess)[0]
sig0 = gaussianML(X)[1]
sig1 = gaussianMAP(X,priormu,sigmaguess)[1]
plt_2dGaussians(mu0, mu1, sig0, sig1)