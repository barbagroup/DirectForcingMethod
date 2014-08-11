from DirectForcingSolver import DirectForcingSolver
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def outside(x, y, R=np.pi/2.):
	return (x-np.pi)**2 + (y-np.pi)**2 >= R**2

def interpolatedField(u, size, factor):
	h = 2*np.pi/(size*factor)
	x = np.arange(h, 2*np.pi+h, h)
	y = np.arange(h/2., 2*np.pi, h)
	newU = np.zeros((size*factor, size*factor))
	n = 0
	for j in xrange(size):
		for i in xrange(size):
			for jj in xrange(factor):
				eta = (jj+1.)/factor
				for ii in xrange(factor):
					xi = (ii+1.)/factor
					J = j*factor+jj
					I = i*factor+ii
					newU[J, I] = (1.-xi)*(1.-eta)*u[j-1,i-1] + (xi)*(1.-eta)*u[j-1,i] + (xi)*(eta)*u[j,i] + (1.-xi)*(eta)*u[j,i-1]
					if not outside(x[I], y[J]):
						newU[J, I] = 0.
	return newU

def plotField(u, size, name):
	h = 2*np.pi/size
	x = np.arange(h, 2*np.pi+h, h)
	y = np.arange(h/2., 2*np.pi, h)
	X, Y = np.meshgrid(x,y)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_wireframe(X, Y, u)
	ax.set_zlim3d(0, 2)
	plt.savefig("%s.png" % (name))

if __name__ == "__main__":
	NT = 20
	START_SIZE = 15

	h = 2*np.pi/START_SIZE
	x = np.arange(h, 2*np.pi+h, h)
	y = np.arange(h/2., 2*np.pi, h)
	X, Y = np.meshgrid(x,y)

	print "-"*80
	print "Direct Forcing solver"
	print "-"*80
	size = START_SIZE
	print size, 'x', size
	solver = DirectForcingSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=0.05, dt=0.0001, order='linear', folder="DF"+str(size), side='outside')
	solver.runSimulation(nt=NT, nsave=NT)
	print " "
	u0 = np.reshape(solver.q[::2]/solver.h, (size, size))
	newU0 = interpolatedField(u0, size, 9)
	plotField(newU0, size*9, str(size))

	size *= 3
	print size, 'x', size
	solver = DirectForcingSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=0.05, dt=0.0001, order='linear', folder="DF"+str(size), side='outside')
	solver.runSimulation(nt=NT, nsave=NT)
	print " "
	u1 = np.reshape(solver.q[::2]/solver.h, (size, size))
	newU1 = interpolatedField(u1, size, 3)
	plotField(newU1, size*3, str(size))
	#u1 = np.reshape(solver.qZeroed[::2]/solver.h, (size, size))
	e10 = la.norm(newU1-newU0)
	print "Difference between 1 and 0:", e10
	print " "

	size *= 3
	print size, 'x', size
	solver = DirectForcingSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=0.05, dt=0.0001, order='linear', folder="DF"+str(size), side='outside')
	solver.runSimulation(nt=NT, nsave=NT)
	print " "
	u2 = np.reshape(solver.q[::2]/solver.h, (size, size))
	plotField(u2, size, str(size))
	#u2 = np.reshape(solver.qZeroed[::2]/solver.h, (size, size))
	e21 = la.norm(u2-newU1)
	print "Difference between 2 and 1:", e21
	print " "

	print "Experimental order of convergence:", (np.log(e21)-np.log(e10))/np.log(3)