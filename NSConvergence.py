from NavierStokesSolver import NavierStokesSolver
import numpy as np
import numpy.linalg as la

NT = 20
START_SIZE = 20

print "-"*80
print "Navier-Stokes solver"
print "-"*80
size = START_SIZE
solver = NavierStokesSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=0.001, folder="NS-grid0")
solver.runSimulation(nt=NT, plot=True, nsave=NT)
u0 = np.reshape(solver.q[::2]/solver.h, (size, size))
solver.exactSolutionTaylorGreen(solver.dt*NT)
uExact = np.reshape(solver.exactSolution[::2]/solver.h, (size, size))
print "Difference between 0 and exact:", la.norm(u0-uExact)
print " "

size *= 3
solver = NavierStokesSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=0.001)
solver.runSimulation(nt=NT, plot=False)
u1 = np.reshape(solver.q[::2]/solver.h, (size, size))[1::3,2::3]
print "Difference between 1 and exact:", la.norm(u1-uExact)
print "Difference between 1 and 0    :", la.norm(u1-u0)
print "Theoretical order of convergence:", (np.log(la.norm(u1-uExact))-np.log(la.norm(u0-uExact)))/np.log(3)
print " "

size *= 3
solver = NavierStokesSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=0.001, folder="NS-grid2")
solver.runSimulation(nt=NT, plot=True, nsave=NT)
u2 = np.reshape(solver.q[::2]/solver.h, (size, size))[4::9,8::9]
print "Difference between 2 and exact:", la.norm(u2-uExact)
print "Difference between 2 and 1    :", la.norm(u2-u1)
print "Theoretical order of convergence:", (np.log(la.norm(u2-uExact))-np.log(la.norm(u1-uExact)))/np.log(3)
print " "

print "Experimental order of convergence:", (np.log(la.norm(u2-u1))-np.log(la.norm(u1-u0)))/np.log(3)