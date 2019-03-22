#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:40:45 2019

@author: xungejiang
"""

"""
Learn more about relevant scipy tools here:
    
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invwishart.html

In Class Exercise:  xx/100  (+ 10 extra credit if submitted before end of day of class)

******************************************************

Goal: Conceptually and Empirically Investigate Bayesian Updating with Conjugate Priors

1) Step through the existing code and gain intuition for the Beta Distribution
    hyperparameters within the context of informative priors and uninformative
    priors.
    
2) Create (code) a similar example with similar visual displays for a Gaussian, 
    Gaussian-Inverse-Wishart conjugate pair. Provide an example and illustration
    when using an informative prior and uninformative prior. Clearly label 
    (in comments) the hyperparameters in the prior. 

********************************************************
"""
# matplotlib inline
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sympy as sp
#import pymc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.stats import invwishart
from scipy.stats import norm
from scipy.special import gamma

from sympy.interactive import printing
printing.init_printing()


# binomial <-> beta, Gaussian <-> Gaussian Inverse Wishart

# Simulate data
np.random.seed(123)

nobs = 10
mu = 0 # True Mean / Sigma of the Data
sigma = 1
Y = np.random.normal(size=nobs)
y_pdf = stats.norm(mu, sigma).pdf(Y)

# Plot the data
fig = plt.figure(figsize=(7,3))
gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1]) 
ax1 = fig.add_subplot(gs[0])

ax1.plot(Y, y_pdf, 'x')

fig.tight_layout()

a1 = 1 # mu 
a2 = 1 # sigma
# Prior Mean
prior_mean = a1
print('Prior mean:', prior_mean)

# Plot the prior
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
X = np.linspace(-1, 1, 100)

ax.plot(X, stats.norm(a1, a2).pdf(X), 'g');

# Cleanup
ax.set(title='Prior Distribution', ylim=(0,12))
ax.legend(['Prior']);

###############################################
# N = small
# Find the hyperparameters of the posterior
a1_hat = a1 * (Y.std()**2 / (nobs * a2**2 + Y.std()**2)) + Y.mean() * (nobs* a2**2 / (nobs*(a2**2) + Y.std()**2))
a2_hat = 1/(1/a2**2 + nobs/Y.std()**2) 

# Posterior Mean
post_mean = a1_hat
print('Posterior Mean (Analytic):', post_mean)

# Plot the analytic posterior after N = 4 Observations
fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(111)
X = np.linspace(-1,1, 1000)
ax.plot(X, stats.norm(a1_hat, a2_hat).pdf(X), 'r');
#####################################################

########################################################
# Simulate data  N = larger
np.random.seed(123)

nobs = 25
Y = np.random.normal(mu, sigma, nobs)

# Find the hyperparameters of the posterior
a1_hat = a1 * (Y.std()**2 / (nobs * a2**2 + Y.std()**2)) + Y.mean() * (nobs* a2**2 / (nobs*(a2**2) + Y.std()**2))
a2_hat = 1/(1/a2**2 + nobs/Y.std()**2) 

# Posterior Mean
post_mean = a1_hat
print('Posterior Mean (Analytic):', post_mean)

# Plot the analytic posterior after N = 4 Observations
#fig = plt.figure(figsize=(10,4))
#ax = fig.add_subplot(111)
#X = np.linspace(0,1, 1000)
ax.plot(X, stats.norm(a1_hat, a2_hat).pdf(X), 'b');

# Plot the prior
ax.plot(X, stats.norm(a1, a2).pdf(X), 'g');

# Cleanup
ax.set(title='Simulate Bayesian Updating non-informative Prior\nPosterior Distribution (Analytic)', ylim=(0,12))
ax.legend(['Posterior (Analytic, n = small)', 'Posterior (Analytic, n = larger)',  'Prior']);


########################################################################
######  Re-run similar simulation with  informative prior  #############
########################################################################

# Simulate data
np.random.seed(123)

# True Mean / Sigma of the Data
# mu = [0, 0]
# k0 = 2
# df/v0 = 10
# psi = [1, 1]
nobs = 10
mu = [0, 0]
k0 = 2
df=10
psi = [1, 1]
sigma = stats.invwishart.rvs(df=df, scale=psi, size=1) / k0
Y = stats.multivariate_normal(mu, [[sigma[0,0], 0], [0,sigma[1,1]]])

########################################################################

a1 = [2, 2] #mu 
a2 = 2      #k0
a3 = 10     #df/v0
a4 = [1, 1] #psi

# Prior Mean
prior_mean = a1
print('Prior mean:', prior_mean)

x1 = np.linspace(-1,1, 1000)
x2 = np.linspace(-1,1, 1000)
X1, X2 = np.meshgrid(x1, x2)
Z = np.empty(X1.shape + (2,))
Z[:, :, 0] = X1
Z[:, :, 1] = X2 # Z is the data

# Plot the prior
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, Y.pdf(Z), cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set(title='Prior Distribution (Gaussian Inverse Wishart)')
plt.show()
###############################################
# N = small
# Find the hyperparameters of the posterior

# update hyperparameters
a1_hat = [(a2*mu[0] + nobs*X1.mean())/(a2+nobs), (a2*mu[1] + nobs*X2.mean())/(a2+nobs)]
a2_hat = a2 + nobs
a3_hat = a3 + nobs
a4_hat = a4 + Z.var() + (a2 * nobs / (a2+nobs)) * (Z.mean() - a1) * (Z.mean() - a1)
sigma_hat = stats.invwishart.rvs(a3_hat, a4_hat, 1) / a2_hat
Y = stats.multivariate_normal(a1_hat, [[sigma_hat[0,0], 0], [0,sigma_hat[1,1]]])

# Posterior Mean
post_mean = a1_hat
print('Posterior Mean (Analytic):', post_mean)

# Plot the analytic posterior after N = 4 Observations
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, Y.pdf(Z), cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set(title='Simulate Bayesian Updating non-informative Prior\nPosterior Distribution (Analytic), n = small')
plt.show()
#####################################################

########################################################
# Simulate data  N = larger

np.random.seed(123)

nobs = 25
mu = [0, 0]
k0 = 2
df=10
psi = [1, 1]
sigma = stats.invwishart.rvs(df=df, scale=psi, size=1) / k0
Y = stats.multivariate_normal(mu, [[sigma[0,0], 0], [0,sigma[1,1]]])

a1_hat = [(a2*mu[0] + nobs*X1.mean())/(a2+nobs), (a2*mu[1] + nobs*X2.mean())/(a2+nobs)]
a2_hat = a2 + nobs
a3_hat = a3 + nobs
a4_hat = a4 + Z.var() + (a2 * nobs / (a2+nobs)) * (Z.mean() - a1) * (Z.mean() - a1)
sigma_hat = stats.invwishart.rvs(a3_hat, a4_hat, 1) / a2_hat
Y = stats.multivariate_normal(a1_hat, [[sigma_hat[0,0], 0], [0,sigma_hat[1,1]]])

# Posterior Mean
post_mean = a1_hat
print('Posterior Mean (Analytic):', post_mean)

# Plot the analytic posterior after N = 4 Observations
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, Y.pdf(Z), cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set(title='Simulate Bayesian Updating Informative Prior\nPosterior Distribution (Analytic), n = larger')
plt.show()

#########################################################################
##################





















