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
	a = y[left]+width/2.
	C2_left = 2*a/(a+h)
	C3_left = -a/(a+2*h)

	right = N-1
	while y[right]-eps > width/2.:
		right-=1
	a = width/2.-y[right]
	C2_right = 2*a/(a+h)
	C3_right = -a/(a+2*h)

	for i in xrange(len(mask)):
		if i<left or i>right:
			mask[i] = 0.

	u = np.zeros(N)
	uExact = np.zeros(N)
	uExact[:] = dpdx/nu/8.*(4*y[:]*y[:]-width**2)

	# matrix
	rows = np.zeros(5*N, dtype=np.int)
	cols = np.zeros(5*N, dtype=np.int)
	vals = np.zeros(5*N, dtype=np.float)
	# rhs
	b = np.zeros(N)

	index = 0

	for i in xrange(N):
		# coefficent of u_{i-2}
		rows[index] = i
		cols[index] = i-2 if i>1 else N+i-2
		if i==left:
			vals[index] = 0.
		elif i==right:
			vals[index] = -C3_right if interp=='quadratic' else 0.
		else:
			vals[index] = 0.
		index+=1

		# coefficient of u_{i-1}
		rows[index] = i
		cols[index] = i-1 if i>0 else N+i-1
		if i==left:
			vals[index] = 0.
		elif i==right:
			vals[index] = -C2_right if interp=='quadratic' else 0.
		else:
			vals[index] = 1.
		index+=1

		rows[index] = i
		cols[index] = i
		vals[index] = 1. if (i==left or i==right) else -2.
		index+=1

		rows[index] = i
		cols[index] = i+1 if i<N-1 else i+1-N
		if i==left:
			vals[index] = -C2_left if interp=='quadratic' else 0.
		elif i==right:
			vals[index] = 0.
		else:
			vals[index] = 1.
		index+=1

		rows[index] = i
		cols[index] = i+2 if i<N-2 else i+2-N
		if i==left:
			vals[index] = -C3_left if interp=='quadratic' else 0.
		elif i==right:
			vals[index] = 0.
		else:
			vals[index] = 0.
		index+=1

		b[i] = 0. if (i==left or i==right) else dpdx/nu*h**2

	A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
	#if N==10:
	#	print A

	#e, _ = la.eig(A.todense())
	#print e

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

	return u*mask, la.norm((u-uExact)*mask)/la.norm(uExact*mask), y[left], y[right]

def two_grid_convergence(start_size, interp_type, folder):
	PATH = '1D-Steady-Quadratic/%s/%s/two_grid' % (interp_type, folder)
	print "%d: " % start_size,
	NUM_GRIDS = 3
	errors = np.zeros(NUM_GRIDS)
	for i in range(NUM_GRIDS):
		size = start_size*(3**i)
		_, errors[i], _, _ = steady_channel_flow(N=size, interp=interp_type, folder=PATH)
	
	print "%1.4f, %1.4f" % (np.log(errors[0]/errors[1])/np.log(3), np.log(errors[1]/errors[2])/np.log(3))
	print errors

	'''
	order_of_convergence = -np.log(errors[-2]/errors[-1])/np.log(mesh_sizes[-2]*1./mesh_sizes[-1])

	first_order = np.array([0.5*errors[0], 0.5*errors[0]*(mesh_sizes[-1]/mesh_sizes[0])**(-1)])
	second_order = np.array([0.5*errors[0], 0.5*errors[0]*(mesh_sizes[-1]/mesh_sizes[0])**(-2)])
	x_coords = np.array([mesh_sizes[0], mesh_sizes[-1]])
	
	if interp_type=="constant":
		TITLE = "Assign the wall velocity to the nearest node"
	elif interp_type=="linear":
		TITLE = "Linear interpolation to the node nearest to the wall"

	plt.loglog(mesh_sizes, errors, 'o-', label='Numerical error (%1.2f)' % (order_of_convergence))
	plt.loglog(x_coords, first_order, label="First-order convergence")
	plt.loglog(x_coords, second_order, label="Second-order convergence")
	plt.xlabel('Mesh size')
	plt.ylabel('Error')
	plt.title(TITLE)
	plt.legend()
	plt.savefig("%s/convergence.png" % (PATH))
	'''

def three_grid_convergence(start_size, interp_type, folder):
	PATH = '1D-Steady-Quadratic/%s/%s/three_grid' % (interp_type, folder)
	print "%d: " % start_size,
	NUM_GRIDS = 4
	u = [[]]*NUM_GRIDS
	diffs = np.zeros(NUM_GRIDS-1)
	y_left  = np.zeros(NUM_GRIDS)
	y_right = np.zeros(NUM_GRIDS)
	for i in range(NUM_GRIDS):
		size = start_size*(3**i)
		u[i], _, y_left[i], y_right[i] = steady_channel_flow(N=size, interp=interp_type, folder=PATH)

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
	plt.clf()

if __name__=="__main__":
	START = 10
	END = 21
	INTERP = "quadratic"
	for size in range(START,END):
		FOLDER = str(size) + '-' + INTERP
		two_grid_convergence(size, INTERP, FOLDER)
		three_grid_convergence(size, INTERP, FOLDER)
		print " "
