import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy.linalg as la
import matplotlib.pyplot as plt
import os, errno

def unsteady_channel_flow(N=20, nu=.125, dpdx=-1., interp='linear', nt=400, dt=0.0125, folder="new"):
	try:
		os.makedirs(folder)
	except OSError as exc:
		if exc.errno == errno.EEXIST and os.path.isdir(folder):
			pass
		else:
			raise
	h = 1./N
	y = np.linspace(-0.5+h/2., 0.5-h/2., N)
	mask = np.ones(N)
	width = 0.8
	left = 0
	while y[left] < -width/2.:
		left+=1
	xi_left = (y[left]+width/2.)/(y[left]+width/2.+h)
	right = N-1
	while y[right] > width/2.:
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

		'''
		if n%10==0:
			plt.ioff()
			plt.plot(y, u, label='Numerical')
			plt.plot(y, uExact, label='Exact')
			plt.legend()
			plt.axis([-0.5,0.5,0,-dpdx/nu/8*width*width*2])
			plt.savefig('output%03d.png' % n)
			plt.clf()
		'''

	plt.ioff()
	plt.plot(y, u, label='Numerical')
	plt.plot(y, uExact, label='Exact')
	plt.legend()
	plt.axis([-0.5,0.5,0,1.2])
	plt.savefig('%s/output%03d.png' % (folder, N))
	plt.clf()

	return y, u, mask

if __name__ == "__main__":
	START_SIZE = 10
	INTERP = 'linear'
	FOLDER = '1D-Unsteady/' + str(START_SIZE) + '-' + INTERP

	print "Interpolation type:", INTERP
	print "Initial mesh size: ", str(START_SIZE)
	
	SIZE = START_SIZE
	y0, u0, umask = unsteady_channel_flow(N=SIZE, interp=INTERP, nt=2000, dt=0.0001, folder=FOLDER)

	SIZE = SIZE*3
	y1, u1, _ = unsteady_channel_flow(N=SIZE, interp=INTERP, nt=2000, dt=0.0001, folder=FOLDER)
	e1 = la.norm(u1[1::3]-u0)

	SIZE = SIZE*3
	y2, u2, _ = unsteady_channel_flow(N=SIZE, interp=INTERP, nt=2000, dt=0.0001, folder=FOLDER)
	e2 = la.norm(u2[4::9]-u1[1::3])

	order_of_convergence = np.log(e2/e1)/np.log(3)
	print "Order of convergence: %f" % order_of_convergence

	SIZE = SIZE*3
	y3, u3, _ = unsteady_channel_flow(N=SIZE, interp=INTERP, nt=2000, dt=0.0001, folder=FOLDER)
	e3 = la.norm(u3[13::27]-u2[4::9])

	order_of_convergence = np.log(e3/e2)/np.log(3)
	print "Order of convergence: %f" % order_of_convergence
	
	plt.ioff()
	plt.clf()
	plt.plot(y0, u0*umask, 'o-', label="%d" % (len(u0)))
	plt.plot(y1[1::3], u1[1::3]*umask, 'o-', label="%d" % (len(u1)))
	plt.plot(y2[4::9], u2[4::9]*umask, 'o-', label="%d" % (len(u2)))
	plt.plot(y3[13::27], u3[13::27]*umask, 'o-', label="%d" % (len(u3)))
	plt.axis([-0.5,0.5,0,1.2])
	plt.legend()
	plt.savefig("%s/solutions.png" % (FOLDER))