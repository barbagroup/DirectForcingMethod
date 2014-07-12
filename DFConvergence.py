from DirectForcingSolver import DirectForcingSolver
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

NT = 10
h = 2*np.pi/10
x = np.arange(h, 2*np.pi+h, h)
y = np.arange(h/2., 2*np.pi, h)
X, Y = np.meshgrid(x,y)

print "-"*80
print "Direct Forcing solver"
print "-"*80
size = 10
solver = DirectForcingSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=1.0, dt=0.001, order='constant', folder="DF10")
solver.runSimulation(nt=NT, nsave=NT)
u0 = np.reshape(solver.q[::2]/solver.h, (size, size))
print " "
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_wireframe(X, Y, u0)
ax.set_zlim3d(0, 2)
plt.savefig("%d.png" % (size))

size = 30
solver = DirectForcingSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=1.0, dt=0.001, order='constant', folder="DF30")
solver.runSimulation(nt=NT, nsave=NT)
u1 = np.reshape(solver.q[::2]/solver.h, (size, size))[1::3,2::3]
print "Difference between 1 and 0:", la.norm(u1-u0)
print " "
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_wireframe(X, Y, u1)
ax.set_zlim3d(0, 2)
plt.savefig("%d.png" % (size))

size = 90
solver = DirectForcingSolver(N=size, alphaExplicit=0., alphaImplicit=1., nu=1.0, dt=0.001, order='constant', folder="DF90")
solver.runSimulation(nt=NT, nsave=NT)
u2 = np.reshape(solver.q[::2]/solver.h, (size, size))[4::9,8::9]
print "Difference between 2 and 1:", la.norm(u2-u1)
print " "
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_wireframe(X, Y, u2)
ax.set_zlim3d(0, 2)
plt.savefig("%d.png" % (size))

print "Experimental order of convergence:", (np.log(la.norm(u2-u1))-np.log(la.norm(u1-u0)))/np.log(3)