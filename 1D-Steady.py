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

def steady_channel_flow(N=20, nu=.125, dpdx=-1., interp='linear', folder="new"):
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
			vals[index] = 1.
		index+=1

		rows[index] = i
		cols[index] = i
		vals[index] = 1. if (i==left or i==right) else -2.
		index+=1

		rows[index] = i
		cols[index] = i+1 if i<N-1 else 0
		if i==left:
			vals[index] = -xi_left if interp=='linear' else 0.
		elif i==right:
			vals[index] = 0.
		else:
			vals[index] = 1.
		index+=1

		b[i] = 0. if (i==left or i==right) else dpdx/nu*h**2

	A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

	#if N==10:
	#	print A

	u, _ = sla.bicgstab(A, b, tol=1e-8)

	plt.ioff()
	plt.clf()
	plt.plot(y, u, 'r', label='Numerical', color='blue')
	plt.plot(y[left:right+1], u[left:right+1], label='Numerical', color='green')
	plt.plot(y[left:right+1], uExact[left:right+1], label='Exact', color='red')
	plt.legend()
	plt.axis([-0.5,0.5,0,-dpdx/nu/8*width*width*1.5])
	plt.savefig('%s/mesh-%d.png' % (folder, N))
	plt.clf()

	return u*mask, (u-uExact)*mask, y[left], y[right]

def two_grid_convergence(start_size, interp_type, num_grids, folder):
	PATH = '1D-Steady/%s/%s/two_grid' % (interp_type, folder)
	errors = [[]]*num_grids
	error_norms = np.zeros(num_grids)
	sizes = np.zeros(num_grids, dtype=int)
	observed_rates = np.zeros(num_grids-1)
	theoretical_rates = np.zeros(num_grids-1)
	y_right = np.zeros(num_grids)
	start = 0
	stride = 1
	for i in range(num_grids):
		size = start_size*(3**i)
		_, errors[i], _, y_right[i] = steady_channel_flow(N=size, interp=interp_type, folder=PATH)
		error_norms[i] = la.norm(errors[i][start::stride])
		start  += 3**i
		stride *= 3
		sizes[i] = size

	h = 0.4-y_right[:]
	np.set_printoptions(6)
	print "{}\t{}".format(sizes[0], h),
	observed_rates[0:] = (np.log(error_norms[1:])-np.log(error_norms[0:-1]))/np.log(1.0/3)
	best_fit_rate = -np.polyfit(np.log(sizes), np.log(error_norms), 1)[0]
	theoretical_rates[0:] = 1 + np.log(h[0:-1]/h[1:])/np.log(3.0)
	np.set_printoptions(3)
	print "\t{}\t{:.3f}\t{}".format(observed_rates, best_fit_rate, theoretical_rates)

def three_grid_convergence(start_size, interp_type, num_grids, folder):
	PATH = '1D-Steady/%s/%s/three_grid' % (interp_type, folder)
	u = [[]]*num_grids
	diffs = np.zeros(num_grids-1)
	sizes = np.zeros(num_grids-1, dtype=int)
	observed_rates = np.zeros(num_grids-2)
	theoretical_rates = np.zeros(num_grids-2)
	y_left  = np.zeros(num_grids)
	y_right = np.zeros(num_grids)
	start0 = 0
	stride0 = 1
	for i in range(num_grids):
		size = start_size*(3**i)
		u[i], _, y_left[i], y_right[i] = steady_channel_flow(N=size, interp=interp_type, folder=PATH)
		if i>0:
			start1  = start0 + 3**(i-1)
			stride1 = stride0*3
			diffs[i-1] = la.norm(u[i][start1::stride1] - u[i-1][start0::stride0])
			sizes[i-1] = size
			start0  = start1
			stride0 = stride1
	
	h = 0.4-y_right[:]
	np.set_printoptions(6)
	print "{}\t{}".format(sizes[0]/3, h),
	observed_rates[0:] = (np.log(diffs[1:])-np.log(diffs[0:-1]))/np.log(1.0/3)
	best_fit_rate = -np.polyfit(np.log(sizes), np.log(diffs), 1)[0]
	theoretical_rates[0:] = 1 + np.log((h[1:-1]-3*h[0:-2])/(h[2:]-3*h[1:-1]))/np.log(3.0)
	np.set_printoptions(3)
	print "\t{}\t{:.3f}\t{}".format(observed_rates, best_fit_rate, theoretical_rates)

if __name__=="__main__":
	START = 10
	END = 21
	INTERP = "linear"
	NUM_GRIDS = 5
	print "\nTwo-grid convergence"
	for size in range(START,END):
		FOLDER = str(size) + '-' + INTERP
		two_grid_convergence(size, INTERP, NUM_GRIDS, FOLDER)
	print "\nThree-grid convergence"
	for size in range(START,END):
		FOLDER = str(size) + '-' + INTERP
		three_grid_convergence(size, INTERP, NUM_GRIDS, FOLDER)