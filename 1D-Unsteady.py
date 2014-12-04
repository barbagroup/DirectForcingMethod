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

y_left_wall  = -0.4
y_right_wall = +0.4
width = y_right_wall - y_left_wall
y_origin = (y_left_wall+y_right_wall)/2.

def unsteady_channel_flow(N=20, nu=.125, dpdx=-1., interp='linear', nt=800, dt=0.001, folder="new"):
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
	
	left = 0
	while y[left]+eps < y_left_wall:
		left+=1
	xi_left = (y[left]-y_left_wall)/(y[left]-y_left_wall+h)
	
	right = N-1
	while y[right]-eps > y_right_wall:
		right-=1
	xi_right = (y_right_wall-y[right])/(y_right_wall-y[right]+h)

	for i in xrange(len(mask)):
		if i<left or i>right:
			mask[i] = 0.

	u = np.zeros(N)
	uExact = np.zeros(N)
	uExact[:] = dpdx/nu/8.*(4*(y[:]-y_origin)*(y[:]-y_origin)-width**2)

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

	return u*mask, y[left], y[right]

def three_grid_convergence(start_size, interp_type, num_grids, folder):
	PATH = '1D-Unsteady/%s/%s/three_grid' % (interp_type, folder)
	NUM_GRIDS = num_grids
	u = [[]]*NUM_GRIDS
	diffs = np.zeros(NUM_GRIDS-1)
	sizes = np.zeros(NUM_GRIDS-1, dtype=int)
	observed_rates = np.zeros(NUM_GRIDS-2)
	theoretical_rates_right = np.zeros(NUM_GRIDS-2)
	theoretical_rates_left  = np.zeros(NUM_GRIDS-2)
	y_left  = np.zeros(NUM_GRIDS)
	y_right = np.zeros(NUM_GRIDS)
	start0 = 0
	stride0 = 1
	for i in range(NUM_GRIDS):
		size = start_size*(3**i)
		u[i], y_left[i], y_right[i] = unsteady_channel_flow(N=size, interp=interp_type, folder=PATH)
		if i>0:
			start1  = start0 + 3**(i-1)
			stride1 = stride0*3
			diffs[i-1] = la.norm(u[i][start1::stride1] - u[i-1][start0::stride0])
			sizes[i-1] = len(u[i])
			start0  = start1
			stride0 = stride1

	h_right = y_right_wall-y_right[:]
	h_left  = y_left[:]-y_left_wall
	np.set_printoptions(6)
	print "{}: {} {},".format(sizes[0]/3, h_left, h_right),
	observed_rates[0:] = (np.log(diffs[1:])-np.log(diffs[0:-1]))/np.log(1.0/3)
	best_fit_rate = -np.polyfit(np.log(sizes), np.log(diffs), 1)[0]
	theoretical_rates_left[0:] = 1 + np.log((h_left[1:-1]-3*h_left[0:-2])/(h_left[2:]-3*h_left[1:-1]))/np.log(3.0)
	theoretical_rates_right[0:] = 1 + np.log((h_right[1:-1]-3*h_right[0:-2])/(h_right[2:]-3*h_right[1:-1]))/np.log(3.0)
	np.set_printoptions(3)
	print "   Expt: {} {:.3f},   Theory: left: {} right: {}".format(observed_rates, best_fit_rate, theoretical_rates_left, theoretical_rates_right)

if __name__=="__main__":
	START = 10
	END = 21
	INTERP = "linear"
	NUM_GRIDS = 4
	print "\nThree-grid convergence"
	for size in range(START,END):
		FOLDER = str(size) + '-' + INTERP
		three_grid_convergence(size, INTERP, NUM_GRIDS, FOLDER)