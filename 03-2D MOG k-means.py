# -*- coding: utf-8 -*-


import numpy as npy
import pylab as pyl
import numpy.matlib as ml
import random

def calc_prob(k,pMiu,pSigma):
    Px = npy.zeros([len(samples.T), k], dtype=float)
    for i in range(k):
        Xshift = npy.mat(X - pMiu[i, :])
        inv_pSigma = npy.mat(pSigma[:, :, i]).I
        coef = pow((2*npy.pi), (len(X[0])/2)) * npy.sqrt(npy.linalg.det(npy.mat(pSigma[:, :, i]))) 
        for j in range(len(samples.T)):
            tmp = (Xshift[j, :] * inv_pSigma * Xshift[j, :].T)
            Px[j, i] = 1.0 / coef * npy.exp(-0.5*tmp)
    return Px    

def distmat(X, Y):
    n = len(X)
    m = len(Y)
    xx = ml.sum(X*X, axis=1)
    yy = ml.sum(Y*Y, axis=1)
    xy = ml.dot(X, Y.T)
    return npy.tile(xx, (m, 1)).T+npy.tile(yy, (n, 1)) - 2*xy
	
def init_params(centers,k):
    pMiu = centers
    pPi = npy.zeros([1,k], dtype=float)
    pSigma = npy.zeros([len(X[0]), len(X[0]), k], dtype=float)
    dist = distmat(X, centers)
    labels = dist.argmin(axis=1)
    for j in range(k):
        idx_j = (labels == j).nonzero()
        pMiu[j] = X[idx_j].mean(axis=0)
        pPi[0, j] = 1.0 * len(X[idx_j]) / len(samples.T)
        pSigma[:, :, j] = npy.cov(npy.mat(X[idx_j]).T)
    return pMiu, pPi, pSigma

mean = [0,10]
cov = [[1,0],[0,100]]
k = 3
samples = npy.random.multivariate_normal(mean,cov,1000).T
X = samples.T;
#Mixtrue of Gauss
labels = npy.zeros(len(X), dtype=int)
centers = npy.array(random.sample(X, k))
iter = 0
Lprev = float('-10000')
pre_esp = 100000
threshold=1e-15
maxiter=500
pMiu, pPi, pSigma = init_params(centers,k)
while iter < maxiter:
    Px = calc_prob(k,pMiu,pSigma)
    pGamma =npy.mat(npy.array(Px) * npy.array(pPi))
    pGamma = pGamma / pGamma.sum(axis=1)
    Nk = pGamma.sum(axis=0) #[1, K]
    pMiu = npy.diagflat(1/Nk) * pGamma.T * npy.mat(X) # Miu=[K, 2]
    pPi = Nk / len(samples.T) #[1, K]
    pSigma = npy.zeros([len(X[0]), len(X[0]), k], dtype=float)
    for j in range(k): 
        Xshift = npy.mat(X) - pMiu[j, :]
        for i in range(len(samples.T)):
            pSigmaK = Xshift[i, :].T * Xshift[i, :]
            pSigmaK = pSigmaK * pGamma[i, j] / Nk[0, j]
            pSigma[:, :, j] = pSigma[:, :, j] + pSigmaK
    labels = pGamma.argmax(axis=1)
    iter = iter + 1
    L = sum(npy.log(npy.mat(Px) * npy.mat(pPi).T))
    cur_esp = L-Lprev
    if cur_esp < threshold:
        break
    if cur_esp > pre_esp:
        break
    pre_esp=cur_esp
    Lprev = L
    print "iter %d esp %lf" % (iter,cur_esp)

pyl.close('all')
colors = npy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
pyl.plot(hold=False)
pyl.hold(True)
labels = npy.array(labels).ravel()
data_colors=[colors[lbl] for lbl in labels]
pyl.scatter(X[:, 0], X[:, 1], c=data_colors, alpha=0.5)
