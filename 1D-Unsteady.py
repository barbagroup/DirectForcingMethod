#!/usr/bin/python
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import errno

def unsteady_channel_flow(N=20, nu=.125, dpdx=-1., interp='linear', nt=400, dt=0.001, folder="new"):
	try:
		os.makedirs(folder)
	except OSError as exc:
		if exc.errno == errno.EEXIST and os.path.isdir(folder):
			pass
		else:
			raise
	eps = 1.e-8
	h = 1./N
	y = np.linspace(-0.5+h/2., 0.5-h/2., N)
	mask = np.ones(N)
	width = 0.8
	
	left = 0
	while y[left]+eps < -width/2.:
		left+=1
	xi_left = (y[left]+width/2.)/(y[left]+width/2.+h)
	
	right = N-1
	while y[right]-eps > width/2.:
		right-=1
	xi_right = (width/2.-y[right])/(width/2.-y[right]+h)

	for i in xrange(len(mask)):
		if i<left or i>right:
			mask[i] = 0.

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

		u, _ = sla.bicgstab(A, b, tol=1e-8)

	plt.ioff()
	plt.clf()
	plt.plot(y, u, 'r', label='Numerical', color='blue')
	plt.plot(y[left:right+1], u[left:right+1], label='Numerical', color='green')
	plt.plot(y[left:right+1], uExact[left:right+1], label='Exact', color='red')
	plt.legend()
	plt.axis([-0.5,0.5,0,-dpdx/nu/8*width*width*1.5])
	plt.savefig('%s/mesh-%d.png' % (folder, N))

	return u*mask, la.norm((u-uExact)*mask)/la.norm(uExact*mask), y[left], y[right]

def three_grid_convergence(start_size, interp_type, folder):
	PATH = '1D-Unsteady/%s/%s/three_grid' % (interp_type, folder)
	print "%d: " % start_size,
	NUM_GRIDS = 4
	u = [[]]*NUM_GRIDS
	diffs = np.zeros(NUM_GRIDS-1)
	y_left  = np.zeros(NUM_GRIDS)
	y_right = np.zeros(NUM_GRIDS)
	for i in range(NUM_GRIDS):
		size = start_size*(3**i)
		u[i], _, y_left[i], y_right[i] = unsteady_channel_flow(N=size, interp=interp_type, folder=PATH)

	diffs[0] = la.norm(u[1][1::3]-u[0])
	diffs[1] = la.norm(u[2][4::9]-u[1][1::3])
	diffs[2] = la.norm(u[3][13::27]-u[2][4::9])

	print "%1.4f, %1.4f" % (np.log(diffs[0]/diffs[1])/np.log(3), np.log(diffs[1]/diffs[2])/np.log(3))
	print y_right
	print diffs

	h = 1./start_size
	y = np.linspace(-0.5+h/2., 0.5-h/2., start_size)
	plt.ioff()
	plt.clf()
	plt.plot(y, u[0], label="grid0")
	plt.plot(y, u[1][1::3], label="grid1")
	plt.plot(y, u[2][4::9], label="grid2")
	plt.plot(y, u[3][13::27], label="grid3")
	plt.axis([-0.5,0.5,0,1.5*0.64])
	plt.legend()
	plt.savefig('%s/three-grid.png' % (PATH))

if __name__=="__main__":
	START = 10
	END = 21
	INTERP = "linear"
	for size in range(START,END):
		FOLDER = str(size) + '-' + INTERP
		three_grid_convergence(size, INTERP, FOLDER)
		print " "
