from DiffusionSolver import DiffusionSolver
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def plotField(u, size):
	h = 2*np.pi/size
	x = np.arange(h, 2*np.pi+h, h)
	y = np.arange(h/2., 2*np.pi, h)
	X, Y = np.meshgrid(x,y)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_wireframe(X, Y, u)
	ax.set_zlim3d(-0.5, 2)
	plt.savefig("%d.png" % (size))

if __name__ == "__main__":
	NT = 20
	START_SIZE = 20

	h = 2*np.pi/START_SIZE
	x = np.arange(h, 2*np.pi+h, h)
	y = np.arange(h/2., 2*np.pi, h)
	X, Y = np.meshgrid(x,y)

	print "-"*80
	print "Direct Forcing solver"
	print "-"*80
	size = START_SIZE
	print size, 'x', size
	solver = DiffusionSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=0.1, order='constant', folder="grid0", side='outside')
	solver.runSimulation(nt=NT, nsave=NT)
	print " "
	#u0 = np.reshape(solver.q[::2]/solver.h, (size, size))
	u0 = np.reshape(solver.qZeroed[::2]/solver.h, (size, size))
	plotField(u0, size)

	size *= 3
	print size, 'x', size
	solver = DiffusionSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=0.1, order='constant', folder="grid1", side='outside')
	solver.runSimulation(nt=NT, nsave=NT)
	#u1 = np.reshape(solver.q[::2]/solver.h, (size, size))
	u1 = np.reshape(solver.qZeroed[::2]/solver.h, (size, size))
	e10 = la.norm(u1[1::3,2::3]-u0)
	print "Difference between 1 and 0:", e10
	print " "
	plotField(u1, size)

	size *= 3
	print size, 'x', size
	solver = DiffusionSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=0.1, order='constant', folder="grid2", side='outside')
	solver.runSimulation(nt=NT, nsave=NT)
	#u2 = np.reshape(solver.q[::2]/solver.h, (size, size))
	u2 = np.reshape(solver.qZeroed[::2]/solver.h, (size, size))
	e21 = la.norm(u2[4::9,8::9]-u1[1::3,2::3])
	print "Difference between 2 and 1:", e21
	print " "
	plotField(u2, size)

	print "Experimental order of convergence:", (np.log(e21)-np.log(e10))/np.log(3)