from NavierStokesSolver import NavierStokesSolver
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy.linalg as la
import matplotlib.pyplot as plt
import os

nu = 0.125
dpdx = -1.

N = 20
ratio = 0.5
xi = ratio/(ratio+1)
width = 1.
h = width/(N-1 + 2*ratio)
y = np.linspace(-0.5*width+ratio*h, 0.5*width-ratio*h, N)
u = np.zeros(N)
uExact = np.zeros(N)
uExact[:] = dpdx/nu/8.*(4*y[:]*y[:]-width**2)

nt = 400
dt = h/4.

# matrix
rows = np.zeros(3*(N-2)+4, dtype=np.int)
cols = np.zeros(3*(N-2)+4, dtype=np.int)
vals = np.zeros(3*(N-2)+4, dtype=np.float)
# rhs
b = np.zeros(N)

row_index = 0
index = 0

rows[index] = row_index
cols[index] = 0
vals[index] = 1.
index+=1

rows[index] = row_index
cols[index] = 1
vals[index] = -xi
index+=1

row_index+=1

for i in xrange(1,N-1):
	rows[index] = row_index
	cols[index] = i-1
	vals[index] = -nu*dt/h**2
	index+=1

	rows[index] = row_index
	cols[index] = i
	vals[index] = (1. + 2.*nu*dt/h**2)
	index+=1

	rows[index] = row_index
	cols[index] = i+1
	vals[index] = -nu*dt/h**2
	index+=1

	row_index+=1

rows[index] = row_index
cols[index] = N-2
vals[index] = -xi
index+=1

rows[index] = row_index
cols[index] = N-1
vals[index] = 1.
index+=1

row_index+=1

A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

for n in xrange(nt):
	for i in xrange(1, N-1):
		b[i] = -dpdx*dt + u[i]

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