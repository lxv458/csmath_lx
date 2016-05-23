# -*- coding: utf-8 -*-

import numpy as npy
import pylab as pyl
from scipy import linalg
from scipy.optimize import leastsq 

def fake_func(p, x):
    f = npy.poly1d(p)
    return f(x)

def residuals(p, y, x):
    ret = y - fake_func(p, x)
    lamda = 0.0001
    ret = npy.append(ret, 0.5*lamda*(npy.sqrt(p**2)**2))
    return ret
    
pyl.close('all')
x1 = npy.linspace(0,2*npy.pi,100)
x2 = npy.linspace(0,2*npy.pi,15)
y2 = npy.sin(x2) + 0.5*npy.random.randn(15)
yx = npy.sin(x1) + 0.5*npy.random.randn(npy.size(x1))
fig1 = pyl.figure()
pyl.plot(x1,npy.sin(x1),'g-',figure='fig1')
pyl.plot(x2,y2,'ro',figure='fig1')
pyl.title('15 point')
f = npy.polyfit(x2,y2,3)
f1 = npy.polyfit(x2,y2,9)
h1 = pyl.plot(x1,npy.polyval(f,x1),'b-',figure='fig1',label = '3 order')
h2 = pyl.plot(x1,npy.polyval(f1,x1),'m-',figure='fig1',label = '9 order')
pyl.legend()

fig2 = pyl.figure()
pyl.plot(x1,npy.sin(x1),'g-',figure='fig2')
pyl.plot(x1,yx,'ro',figure='fig2')
fx = npy.polyfit(x1,yx,9)
pyl.plot(x1,npy.polyval(fx,x1),'b-',figure='fig2')
pyl.title('100 point')

fig3 = pyl.figure()
pyl.plot(x1,npy.sin(x1),'g-',figure='fig3')
pyl.plot(x1,yx,'ro',figure='fig3')
pyl.title('100 point with regulation')
fr = x1.repeat(10)
fr.shape = 100,10
fr = fr.T
for i in range(10):
    fr[i] = fr[i]**i

w = npy.dot(npy.dot(yx.T,fr.T),linalg.inv(npy.dot(fr,fr.T)-npy.eye(10)*npy.exp(-npy.log(18))))
pyl.plot(x,npy.polyval(w[9:-11:-1],x),'b-',figure='fig3')

fig4 = pyl.figure()
pyl.plot(x,npy.sin(x),'g-',figure='fig4')
pyl.plot(x2,y2,'ro',figure='fig4')
pyl.title('15 point with regulation')
fr1 = x2.repeat(10)
fr1.shape = 15,10
fr1 = fr1.T
for i in range(10):
    fr1[i] = fr1[i]**i

w1 = npy.dot(npy.dot(y2.T,fr1.T),linalg.inv(npy.dot(fr1,fr1.T)-npy.eye(10)*npy.exp(-npy.log(18))))
pyl.plot(x,npy.polyval(w1[9:-11:-1],x),'b-',figure='fig4',label = 'regulate with matrix analysis')
plsq = leastsq(residuals, npy.random.randn(10), args=(y2, x2))
y2=fake_func(plsq[0], x)
pyl.plot(x,y2,'m-',figure='fig4',label = 'regulate with scipy.leatsq')
pyl.legend()
