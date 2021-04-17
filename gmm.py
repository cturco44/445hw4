"""
EECS 445 - Introduction to Machine Learning
Winter 2021 - HW4 - gmm.py
The gmm function takes in as input a data matrix X and a number of gaussians in
the mixture model

The implementation assumes that the covariance matrix is shared and is a
spherical diagonal covariance matrix

If you get this ImportError
    ImportError: cannot import name 'logsumexp' from 'scipy.special',
you may need to update your scipy install to >= 0.19.0
"""

from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp
import numpy as np
import math

import matplotlib.pyplot as plt


def calc_logpdf(x, mean, cov):
    """Return log probability density."""
    x = multivariate_normal.logpdf(x, mean=mean, cov=cov)
    return x


def gmm(trainX, num_K, num_iter=10, plot=False):
    """Fit a gaussian mixture model on trainX data with num_K clusters.

    trainX is a NxD matrix containing N datapoints, each with D features
    num_K is the number of clusters or mixture components
    num_iter is the maximum number of EM iterations run over the dataset

    Description of other variables:
        - mu, which is KxD, the coordinates of the means
        - pk, which is Kx1 and represents the cluster proportions
        - zk, which is NxK, has at each z(n,k) the probability that the nth
          data point belongs to cluster k, specifying the cluster associated
          with each data point
        - si2 is the estimated (shared) variance of the data
        - BIC is the Bayesian Information Criterion (smaller BIC is better)
    """
    N = trainX.shape[0]
    D = trainX.shape[1]

    if num_K >= N:
        print("You are trying too many clusters")
        raise ValueError
    if plot and D != 2:
        print("Can only visualize if D = 2")
        raise ValueError

    si2 = 5  # Initialization of variance
    pk = np.ones((num_K, 1)) / num_K  # Uniformly initialize cluster proportions
    mu = np.random.randn(num_K, D)  # Random initialization of clusters
    zk = np.zeros(
        [N, num_K]
    )  # Matrix containing cluster membership probability for each point

    if plot:
        plt.ion()
        fig = plt.figure()
    for iter in range(0, num_iter):
        """Iterate through one loop of the EM algorithm."""
        if plot:
            plt.clf()
            xVals = trainX[:, 0]
            yVals = trainX[:, 1]
            x = np.linspace(np.min(xVals), np.max(xVals), 500)
            y = np.linspace(np.min(yVals), np.max(yVals), 500)
            X, Y = np.meshgrid(x, y)
            pos = np.array([X.flatten(), Y.flatten()]).T
            plt.scatter(xVals, yVals, color="black")
            pdfs = []
            for k in range(num_K):
                rv = multivariate_normal(mu[k], si2)
                pdfs.append(rv.pdf(pos).reshape(500, 500))
            pdfs = np.array(pdfs)
            plt.contourf(X, Y, np.max(pdfs, axis=0), alpha=0.8)
            plt.pause(0.01)

        """
        E-Step
        In the first step, we find the expected log-likelihood of the data
        which is equivalent to:
        Finding cluster assignments for each point probabilistically
        In this section, you will calculate the values of zk(n,k) for all n and
        k according to current values of si2, pk and mu
        """
        # TODO: Implement the E-step
        cov = si2 * np.identity(D)
        for ni in range(0,N):
            for ki in range(0,num_K):
                gamma = pk[ki][0]
                x = trainX[ni]
                temp_mu = mu[ki]
                normal_value = multivariate_normal.pdf(x, mean=temp_mu, cov=cov)
                zk[ni][ki] = gamma * normal_value
            sum_across_row = np.sum(zk[ni,:])
            zk[ni,:] = zk[ni,:] / sum_across_row
        
        """
        M-step
        Compute the GMM parameters from the expressions which you have in the spec
        """

        
        # TODO
        for ki in range(0,num_K):
            # Estimate new value of pk
            col = zk[:,ki]
            pk[ki] = np.sum(col) / N

        # Estimate new value for means
        # TODO
        for ki in range(0,num_K):
            col = zk[:,ki]
            prob_sum = np.sum(col)
            col = col.reshape(col.shape[0], 1)
            intermediate_matrix = col * trainX
            final_mean = np.sum(intermediate_matrix, axis = 0)
            final_mean = final_mean / prob_sum
            mu[ki] = final_mean
        
        # Estimate new value for sigma^2
        # TODO
        outer_sum = 0
        for it in range(0, N):
            for kit in range(0, num_K):
                temp_zk = zk[it][kit]
                mat = trainX[it]
                mat = mat.reshape(mat.shape[0], 1)
                mat_tranpose = mat.T
                temp = (temp_zk * np.matmul(mat_tranpose, mat))
                outer_sum += temp[0][0]
        divisor = N * D * 1.0
        si2 = outer_sum / divisor

    if plot:
        plt.ioff()
        plt.savefig('visualize_clusters.png')
    # Computing the expected log-likelihood of data for the optimal parameters computed
    # TODO
    # Compute the BIC for the current clustering
    cov = si2 * np.identity(D)
    outer_sum = 0
    for it in range(0, N):
        inner_sum = 0
        for jit in range(0, num_K):
            pk_value = pk[jit][0]
            temp_mu = mu[jit]
            x = trainX[it]
            normal_value = multivariate_normal.pdf(x, temp_mu, cov)
            inner_sum += (pk_value * normal_value)
        outer_sum += np.log(inner_sum)
    num_parameters = num_K - 1 + 1 + D
    BIC = (num_parameters * np.log(N)) - (2 * np.log(outer_sum))  # TODO: calculate BIC


    return mu, pk, zk, si2, BIC
