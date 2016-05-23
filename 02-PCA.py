# -*- coding: utf-8 -*-

import numpy as npy
import pylab as pyl
import itertools as itl

def mean_center(X):
    (rows, cols) = npy.shape(X)
    new_X = npy.zeros((rows, cols), float)
    _averages = npy.average(X, 0)
    for row in range(rows):
        new_X[row, 0:cols] = X[row, 0:cols] - _averages[0:cols]
    return new_X

def standardization(X):
    (rows, cols) = npy.shape(X)
    new_X = npy.zeros((rows, cols))
    _STDs = npy.std(X, 0)
    for value in _STDs:
        if value == 0: raise ZeroDivisionError, 'division by zero, cannot proceed'
    for row in range(rows):
        new_X[row, 0:cols] = X[row, 0:cols] / _STDs[0:cols]
    return new_X

def PCA_svd(X, standardize=False):
    X = mean_center(X)
    if standardize:
        X = standardization(X)
    (rows, cols) = npy.shape(X)
    [U, S, V] = npy.linalg.svd(X) # NOTE!,this line is time consuming which is not allowed on laptop
    if npy.shape(S)[0] < npy.shape(U)[0]: U = U[:, 0:npy.shape(S)[0]]
    Scores = U * S 
    Loadings = V 
    variances = S**2 / cols
    variances_sum = sum(variances)
    explained_var = variances / variances_sum
    return Scores, Loadings, explained_var


FileName='optdigits-orig.wdep'
MAT = []
X = []
for i in open(FileName):
    if not i:
            break
    i = i.strip('\n')
    if len(i) < 5:
        number = int(i)
        if number == 3:
            MAT.append(X)
        X=[]
    else:
        for str in i:
            X.append(int(str))
            
            
data = npy.array(npy.matrix(MAT).T)
T, P, explained_var = PCA_svd(data)
pyl.plot(T[:,0],T[:,1])
