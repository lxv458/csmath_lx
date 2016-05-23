# -*- coding: utf-8 -*-


import numpy as npy
import scipy.linalg as lng
import scipy as sp

data_1= [1.3,2.4,1.0,2.6,7.3,2.9,4.8,5.6,7.3]
obs_1 = [13.42,1.43,13.77,30.25,11.93,3.94,18.39,2.94,9.74]
a0=10.0
b0=0.5
y_init = npy.zeros(len(data_1))
for x_i in range(0,len(data_1)):
    y_init[x_i] = a0*npy.exp(-b0*data_1[x_i])
Ndata = len(obs_1)
Nparams = 2
n_iters = 50
lamda = 0.01
updateJ = 1
a_est = a0
b_est = b0
i = 0
obs_est = npy.zeros(len(data_1))
obs_est_lm = npy.zeros(len(data_1))
while i < n_iters:
    if updateJ == 1:
            J=npy.zeros([Ndata,Nparams])
            for j in range(0,len(data_1)):
                J[j][0] = npy.exp(-b_est*data_1[j])
                J[j][1] = -a_est*data_1[j]*npy.exp(-b_est*data_1[j])
            for y_i in range(len(data_1)):
                obs_est[y_i] = a_est*npy.exp(-b_est*data_1[y_i])
            d = obs_1 - obs_est
            H = sp.dot(J.T,J)
            if i == 0:
                e = npy.sum(d*d)
    H_lm = H + (lamda*npy.eye(Nparams))
    g = sp.dot(J.T,d)
    dp = sp.dot(lng.inv(H_lm),g)
    a_lm = a_est+dp[0]
    b_lm = b_est+dp[1]
    for y_lm_i in range(len(data_1)):
        obs_est_lm[y_lm_i] = a_lm*npy.exp(-b_lm*data_1[y_lm_i])
    d_lm = obs_1-obs_est_lm
    e_lm = npy.sum(d_lm*d_lm)
    if e_lm < e:
        lamda = lamda/10
        a_est = a_lm
        b_est = b_lm
        e=e_lm
        updateJ = 1
    else:
        updateJ = 0
        lamda = lamda*10
    i = i + 1
    print "%f\t%f" %(a_est, b_est)
print "best_parameter_1=%f; best_parameter_2=%f" %(a_est, b_est)
