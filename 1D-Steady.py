from NavierStokesSolver import NavierStokesSolver
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy.linalg as la
import matplotlib.pyplot as plt
import os

nu = 0.125
dpdx = -1.
interp = 'constant'

N = 6
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
		vals[index] = 0. if interp=='constant' else -xi_right
	else:
		vals[index] = 1.
	index+=1

	rows[index] = i
	cols[index] = i
	vals[index] = 1. if (i==left or i==right) else -2.
	index+=1

	rows[index] = i
	cols[index] = i+1 if i<N-1 else 0
	if i==left:
		vals[index] = 0. if interp=='constant' else -xi_left
	elif i==right:
		vals[index] = 0.
	else:
		vals[index] = 1.
	index+=1

	b[i] = 0. if (i==left or i==right) else dpdx/nu*h**2

A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

#e, _ = la.eig(A.todense())
#print e

u, _ = sla.bicgstab(A, b)

plt.ioff()
plt.plot(y, u, 'r', label='Numerical')
plt.plot(y[left:right+1], u[left:right+1], label='Numerical')
plt.plot(y[left:right+1], uExact[left:right+1], label='Exact')
plt.legend()
plt.axis([-0.5,0.5,0,-dpdx/nu/8*width*width*1.5])
plt.savefig('output.png')