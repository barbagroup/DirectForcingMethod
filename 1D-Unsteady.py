import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy.linalg as la
import matplotlib.pyplot as plt
import os

nu = 0.125
dpdx = -1.
interp = 'linear'

N = 20
h = 1./N
y = np.linspace(-0.5+h/2., 0.5-h/2., N)
width = 0.8
left = 0
while y[left] < -width/2.:
	left+=1
xi_left = (y[left]+width/2.)/(y[left]+width/2.+h)
right = N-1
while y[right] > width/2.:
	right-=1
xi_right = (width/2.-y[right])/(width/2.-y[right]+h) 

u = np.zeros(N)
uExact = np.zeros(N)
uExact[:] = dpdx/nu/8.*(4*y[:]*y[:]-width**2)

nt = 400
dt = h/4.

# matrix
rows = np.zeros(3*N, dtype=np.int)
cols = np.zeros(3*N, dtype=np.int)
vals = np.zeros(3*N, dtype=np.float)
# rhs
b = np.zeros(N)

index = 0

for i in xrange(N):
	rows[index] = i
	cols[index] = i-1 if i>0 else N-1
	if i==left:
		vals[index] = 0.
	elif i==right:
		vals[index] = -xi_right if interp=='linear' else 0.
	else:
		vals[index] = -nu*dt/h**2
	index+=1

	rows[index] = i
	cols[index] = i
	vals[index] = 1. if (i==left or i==right) else (1. + 2.*nu*dt/h**2)
	index+=1

	rows[index] = i
	cols[index] = i+1 if i<N-1 else 0
	if i==left:
		vals[index] = -xi_left if interp=='linear' else 0.
	elif i==right:
		vals[index] = 0.
	else:
		vals[index] = -nu*dt/h**2
	index+=1

A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

for n in xrange(nt):
	for i in xrange(1, N-1):
		b[i] = 0. if (i==left or i==right) else -dpdx*dt + u[i]

	#u, _ = sla.cg(A, b)
	u, _ = sla.bicgstab(A, b)

	if n%10==0:
		plt.ioff()
		plt.plot(y, u, label='Numerical')
		plt.plot(y, uExact, label='Exact')
		plt.legend()
		plt.axis([-0.5,0.5,0,-dpdx/nu/8*width*width*2])
		plt.savefig('output%03d.png' % n)
		plt.clf()