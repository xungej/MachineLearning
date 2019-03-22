#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 14:59:58 2019

@author: xungejiang
"""

import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal as mvn
import math

from matplotlib import pyplot as plt

# sample 2d data
np.random.seed(seed=123)
data1 = np.array([[1, 1.5, 1.2, 1.2, .9, .8, 1, 2.3, 2.1, 2, 3, 2.5, 3]])                                
data2 = np.array([[1.5, 1.2, 1.2, .9, .7, .8, 2.3, 2.1, 2, 3, 2.5, 3, 2.1]])
data = np.concatenate((data1, data2), axis=0).T

def gaussian_2d(x, y, x0, y0, xsig, ysig):
    return (1/(2*math.pi*(xsig+ysig)**(0.5)))*np.exp(-0.5*(((x-x0) / xsig)**2 + ((y-y0) / ysig)**2))

# plot function 
def plt_2dGaussians(data, mu, sig, m):
    delta = 0.025
    x = np.arange(-2.0, 5.0, delta)
    y = np.arange(-2.0, 5.0, delta)
    X, Y = np.meshgrid(x, y)
    
    plt.clf()
    plt.figure(figsize=(10,5))
    Z = dict()
    CS = dict()
    # plot raw data
    plt.plot(data[:, 0], data[:, 1], "r*")
    for i in range(m): 
        Z[i] = gaussian_2d(X, Y, mu[i][0], mu[i][1], sig[i][0].sum(), sig[i][1].sum())
        CS[i] = plt.contour(X, Y, Z[i])
        plt.clabel(CS[i], inline=1, fontsize=10)
    plt.title('Learned Gaussian Contours')
    plt.show()

def EM(X, m, epsilon):
    n, p = X.shape
    # dimension - 2d
    l = 2
    
    # initalize - initial guesses for parameters
    # mu: m*l matrix (m gaussians and 2 dimensions)
    mu = np.random.random((m, l)) 
    # sig: l*l matrix for each m => m*l vectors 
    sig = np.random.random((m, l))
    # prior, evenly divided for initial value 
    prior = np.ones(m) / m
    
    old_LL = np.inf  
    max_iter = 100
    
    for i in range(max_iter): 
        # E-step - get probability of each 
        E = np.zeros((m, n))
        for j in range(m):
            for i in range(n):
                E[j, i] = prior[j] * mvn(mu[j], sig[j]).pdf(X[i])
        # expected
        E = E / np.sum(E, axis=0) 
        E = E.T     

        # M-step - Create new mu, sig, prior vectors
        mu = []
        sigma = [] 
        prior = [] 
        for j in range(m):
            # sum by column
            rk = np.sum(E[:,j],axis=0) 

            #update mu 
            mu_e = np.sum(X * E[:, j].reshape(len(X),1), axis=0) 
            mu_e = mu_e / rk
            mu.append(mu_e)
        
            # update sigma
            result = (np.array(E[:, j]).reshape(len(X),1)*(X-mu_e)).T
            sig_e = np.dot(result,(X-mu_e))/rk
            sigma.append(sig_e)
    
            # update prior
            prior_e = rk / np.sum(E)
            prior.append(prior_e)
        
        # plot learned parameters 
        plt_2dGaussians(X, mu, sigma, m)
        
        # update complete log likelihoood
        new_LL = 0 
        for j in range(m): 
            for i in range(n):
                new_LL = np.log(prior[j] * mvn(mu[j], sigma[j]).pdf(X[i]))
        if np.abs(new_LL - old_LL) < epsilon: break
        old_LL = new_LL
        
    return mu, sigma, prior

if __name__ == '__main__':
    # call the EM algorithm, set m gaussian
    m = 2
    mu_result, sigma_result, prior_result = EM(data, m, 0.001)
