# -*- coding: utf-8 -*-


import numpy as npy
import pylab as pyl
import cvxopt
import cvxopt.solvers
from numpy import linalg

def project(X,self_a,self_sv_y,self_sv):
    y_predict = npy.zeros(len(X))
    for i in range(len(X)):
        s = 0
        for a, sv_y, sv in zip(self_a, self_sv_y, self_sv):
            s += a * sv_y * gaussian_kernel(X[i], sv)
        y_predict[i] = s
    return (y_predict + self_b)

def predict(X,self_a,self_sv_y,self_sv):
    return npy.sign(project(X,self_a,self_sv_y,self_sv))

def gaussian_kernel(x, y, sigma=5.0):
    return npy.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def svm_fit(X, Y):
    n_samples, n_features = X.shape
    K = npy.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = gaussian_kernel(X[i], X[j])
    P = cvxopt.matrix(npy.outer(Y,Y) * K)
    q = cvxopt.matrix(npy.ones(n_samples) * -1)
    A = cvxopt.matrix(Y, (1,n_samples))
    b = cvxopt.matrix(0.0)
    G = cvxopt.matrix(npy.diag(npy.ones(n_samples) * -1))
    h = cvxopt.matrix(npy.zeros(n_samples))
    # solve QP problem
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    # Lagrange multipliers
    a = npy.ravel(solution['x'])
    # Support vectors have non zero lagrange multipliers
    sv = a > 1e-5
    ind = npy.arange(len(a))[sv]
    self_a = a[sv]
    self_sv = X[sv]
    self_sv_y = Y[sv]
    print "%d support vectors out of %d points" % (len(self_a), n_samples)
    # Intercept
    self_b = 0
    for n in range(len(self_a)):
        self_b += self_sv_y[n]
        self_b -= npy.sum(self_a * self_sv_y * K[ind[n],sv])
    self_b /= len(self_a)
    return self_a,self_sv,self_sv_y,self_b
    

#data
mean1 = [-1, 2]
mean2 = [1, -1]
mean3 = [4, -4]
mean4 = [-4, 4]
cov = [[1.0,0.8], [0.8, 1.0]]
X1 = npy.random.multivariate_normal(mean1, cov, 50)
X1 = npy.vstack((X1, npy.random.multivariate_normal(mean3, cov, 50)))
Y1 = npy.ones(len(X1))
X2 = npy.random.multivariate_normal(mean2, cov, 50)
X2 = npy.vstack((X2, npy.random.multivariate_normal(mean4, cov, 50)))
Y2 = npy.ones(len(X2)) * -1

#train and test
X1_train = X1[:90]
Y1_train = Y1[:90]
X2_train = X2[:90]
Y2_train = Y2[:90]
X_train = npy.vstack((X1_train, X2_train))
Y_train = npy.hstack((Y1_train, Y2_train))
X1_test = X1[90:]
Y1_test = Y1[90:]
X2_test = X2[90:]
Y2_test = Y2[90:]
X_test = npy.vstack((X1_test, X2_test))
Y_test = npy.hstack((Y1_test, Y2_test))

#svm
self_a,self_sv,self_sv_y,self_b = svm_fit(X_train, Y_train)
Y_predict = predict(X_test,self_a,self_sv_y,self_sv)
correct = npy.sum(Y_predict == Y_test)
print "%d out of %d predictions correct" % (correct, len(Y_predict))
pyl.plot(X1_train[:,0], X1_train[:,1], "ro")
pyl.plot(X2_train[:,0], X2_train[:,1], "yo")
pyl.scatter(self_sv[:,0], self_sv[:,1], s=100, c="g")
X1, X2 = npy.meshgrid(npy.linspace(-6,6,50), npy.linspace(-6,6,50))
X = npy.array([[x1, x2] for x1, x2 in zip(npy.ravel(X1), npy.ravel(X2))])
Z = project(X ,self_a, self_sv_y, self_sv).reshape(X1.shape)
pyl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
pyl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
pyl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')
pyl.axis("tight")
pyl.show()
