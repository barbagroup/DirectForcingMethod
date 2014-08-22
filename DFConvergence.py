from DirectForcingSolver import DirectForcingSolver
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.colors import LogNorm

def plotField(u, size, name, folder):
	h = 2*np.pi/size
	x = np.arange(h, 2*np.pi+h, h)
	y = np.arange(h/2., 2*np.pi, h)
	X, Y = np.meshgrid(x,y)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_wireframe(X, Y, u)
	ax.set_zlim3d(0, 2)
	fig.savefig("%s/%d.png" % (folder, name))

def plotDiff(u2, u1, size, name, folder):
	h = 2*np.pi/size
	x = np.arange(h, 2*np.pi+h, h)
	y = np.arange(h/2., 2*np.pi, h)
	X, Y = np.meshgrid(x,y)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	CS = ax.pcolor(X, Y, abs(u2-u1), norm=LogNorm(vmin=1e-10, vmax=1))
	fig.gca().set_aspect('equal', adjustable='box')
	fig.colorbar(CS)
	fig.savefig("%s/diff%d.png" % (folder, name))

if __name__ == "__main__":
	NT = 20
	START_SIZE = 15
	ORDER = 'linear'
	FOLDER = str(START_SIZE)+'-'+ORDER

	h = 2*np.pi/START_SIZE
	x = np.arange(h, 2*np.pi+h, h)
	y = np.arange(h/2., 2*np.pi, h)
	X, Y = np.meshgrid(x,y)

	print "-"*80
	print "Direct Forcing solver"
	print "-"*80
	size = START_SIZE
	print size, 'x', size
	solver = DirectForcingSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=0.0001, order=ORDER, folder=FOLDER+"/DF"+str(size), side='outside', coarsest=START_SIZE)
	solver.runSimulation(nt=NT, nsave=NT)
	umask, vmask = solver.createMask()
	print " "
	u0 = np.reshape(solver.q[::2]/solver.h, (size, size))
	v0 = np.reshape(solver.q[1::2]/solver.h, (size, size))
	#plotField(u0, size)
	plotField(u0*umask, size, size, FOLDER)

	size *= 3
	print size, 'x', size
	solver = DirectForcingSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=0.0001, order=ORDER, folder=FOLDER+"/DF"+str(size), side='outside', coarsest=START_SIZE)
	solver.runSimulation(nt=NT, nsave=NT)
	u1 = np.reshape(solver.q[::2]/solver.h, (size, size))
	v1 = np.reshape(solver.q[1::2]/solver.h, (size, size))
	e10u = la.norm((u1[1::3,2::3]-u0)*umask)
	e10v = la.norm((v1[2::3,1::3]-v0)*vmask)
	print "Difference between 1 and 0 (u):", e10u
	print "Difference between 1 and 0 (v):", e10v
	print " "
	plotField(u1[1::3,2::3]*umask, START_SIZE, size, FOLDER)
	plotDiff(u1[1::3,2::3]*umask, u0*umask, START_SIZE, size, FOLDER)

	size *= 3
	print size, 'x', size
	solver = DirectForcingSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=0.0001, order=ORDER, folder=FOLDER+"/DF"+str(size), side='outside', coarsest=START_SIZE)
	solver.runSimulation(nt=NT, nsave=NT)
	u2 = np.reshape(solver.q[::2]/solver.h, (size, size))
	v2 = np.reshape(solver.q[1::2]/solver.h, (size, size))
	e21u = la.norm((u2[4::9,8::9]-u1[1::3,2::3])*umask)
	e21v = la.norm((v2[8::9,4::9]-v1[2::3,1::3])*vmask)
	print "Difference between 2 and 1 (u):", e21u
	print "Difference between 2 and 1 (v):", e21v
	print " "
	plotField(u2[4::9,8::9]*umask, START_SIZE, size, FOLDER)
	plotDiff(u2[4::9,8::9]*umask, u1[1::3,2::3]*umask, START_SIZE, size, FOLDER)

	print "Experimental order of convergence (u):", (np.log(e21u)-np.log(e10u))/np.log(3)
	print "Experimental order of convergence (v):", (np.log(e21v)-np.log(e10v))/np.log(3)

	size *= 3
	print size, 'x', size
	solver = DirectForcingSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=0.05, dt=0.0001, order=ORDER, folder=FOLDER+"/DF"+str(size), side='outside', coarsest=START_SIZE)
	solver.runSimulation(nt=NT, nsave=NT)
	u3 = np.reshape(solver.q[::2]/solver.h, (size, size))
	v3 = np.reshape(solver.q[1::2]/solver.h, (size, size))
	e32u = la.norm((u3[13::27,26::27]-u2[4::9,8::9])*umask)
	e32v = la.norm((v3[26::27,13::27]-v2[8::9,4::9])*vmask)
	print "Difference between 2 and 1 (u):", e32u
	print "Difference between 2 and 1 (v):", e32v
	print " "
	plotField(u3[13::27,26::27]*umask, START_SIZE, size, FOLDER)
	plotDiff(u3[13::27,26::27]*umask, u2[4::9,8::9]*umask, START_SIZE, size, FOLDER)

	print "Experimental order of convergence (u):", (np.log(e32u)-np.log(e21u))/np.log(3)
	print "Experimental order of convergence (v):", (np.log(e32v)-np.log(e21v))/np.log(3)