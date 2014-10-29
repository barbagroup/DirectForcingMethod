from NavierStokesSolver import NavierStokesSolver
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy.linalg as la
import matplotlib.pyplot as plt
import os

def outside(x, y, R=np.pi/2.):
	return (x-np.pi)**2 + (y-np.pi)**2 >= R**2

def inside(x, y, R=np.pi/2.):
	return (x-np.pi)**2 + (y-np.pi)**2 <= R**2

def pointOfIntersectionX(xLeft, xRight, y):
	x0 = np.pi + np.sqrt((np.pi/2.)**2 - (y-np.pi)**2)
	x1 = np.pi - np.sqrt((np.pi/2.)**2 - (y-np.pi)**2)
	if xLeft <= x0 and x0 <=xRight:
		return x0
	else:
		return x1

def pointOfIntersectionY(yBottom, yTop, x):
	y0 = np.pi + np.sqrt((np.pi/2.)**2 - (x-np.pi)**2)
	y1 = np.pi - np.sqrt((np.pi/2.)**2 - (x-np.pi)**2)
	if yBottom <= y0 and y0 <=yTop:
		return y0
	else:
		return y1

class FadlunEtAlSolver(NavierStokesSolver):
	def __init__(self, N=4, alphaImplicit=1., alphaExplicit=0., gamma=1., zeta=0., nu=0.01, dt=-1.0, folder=".", order='linear', side='outside', coarsest=15):
		NavierStokesSolver.__init__(self, N, alphaImplicit, alphaExplicit, gamma, zeta, nu, dt, folder)
		self.order = order
		self.side = side
		self.coarsest = coarsest

	def initVecs(self):
		NavierStokesSolver.initVecs(self)
		N = self.N
		self.qZeroed = np.zeros(2*N*N)
		self.tagsX   = -np.ones(2*N*N, dtype=np.int)
		self.coeffsX = np.zeros(2*N*N)
		self.tagsY   = -np.ones(2*N*N, dtype=np.int)
		self.coeffsY = np.zeros(2*N*N)
		self.xu = -np.zeros(N+1)
		self.yu = -np.zeros(N+1)
		self.xv = -np.zeros(N+1)
		self.yv = -np.zeros(N+1)
		self.initCoords()
		self.tagPoints()
		self.net_flux = np.zeros(0)

	def initFluxes(self):
		h = self.h
		N = self.N
		row_index = 0
		for j in xrange(N):
			for i in xrange(N):
				# u
				x = (i+1)*h
				y = (j+0.5)*h
				self.q[row_index] = 1.0*h
				row_index+=1

				# v
				x = (i+0.5)*h
				y = (j+1)*h
				self.q[row_index] = 0.0*h
				row_index+=1

	def initCoords(self):
		N = self.N
		h = self.h
		index = 0
		for j in xrange(N):
			self.yu[j] = (j+0.5)*h
			self.yv[j] = (j+1)*h
			for i in xrange(N):
				self.xu[i] = (i+1)*h
				self.xv[i] = (i+0.5)*h
		self.xu[N] = (N+1)*h
		self.yu[N] = (N+0.5)*h
		self.xv[N] = (N+0.5)*h
		self.yv[N] = (N+1)*h
		#print self.xu
		#print self.yu
		#print self.xv
		#print self.yv

	def tagPoints(self):
		if self.side =='outside':
			self.tagOutsidePoints()
		if self.side =='inside':
			self.tagInsidePoints()
		self.plotTaggedPoints()

	def tagOutsidePoints(self):
		N = self.N
		h = self.h
		index = 0
		for j in xrange(N):
			for i in xrange(N):
				# tagsX
				if outside(self.xu[i], self.yu[j]) and not outside(self.xu[i-1], self.yu[j]):
					x = pointOfIntersectionX(self.xu[i-1], self.xu[i], self.yu[j])
					self.tagsX[index] = index+2
					self.coeffsX[index] = (self.xu[i]-x)/(self.xu[i+1]-x)
				elif outside(self.xu[i], self.yu[j]) and not outside(self.xu[i+1], self.yu[j]):
					x = pointOfIntersectionX(self.xu[i], self.xu[i+1], self.yu[j])
					self.tagsX[index] = index-2
					self.coeffsX[index] = (self.xu[i]-x)/(self.xu[i-1]-x)

				# tagsY
				if outside(self.xu[i], self.yu[j]) and not outside(self.xu[i], self.yu[j-1]):
					y = pointOfIntersectionY(self.yu[j-1], self.yu[j], self.xu[i])
					self.tagsY[index] = index+2*N
					self.coeffsY[index] = (self.yu[j]-y)/(self.yu[j+1]-y)
				elif outside(self.xu[i], self.yu[j]) and not outside(self.xu[i], self.yu[j+1]):
					y = pointOfIntersectionY(self.yu[j], self.yu[j+1], self.xu[i])
					self.tagsY[index] = index-2*N
					self.coeffsY[index] = (self.yu[j]-y)/(self.yu[j-1]-y)

				index+=1

				# tagsX
				if outside(self.xv[i], self.yv[j]) and not outside(self.xv[i-1], self.yv[j]):
					x = pointOfIntersectionX(self.xv[i-1], self.xv[i], self.yv[j])
					self.tagsX[index] = index+2
					self.coeffsX[index] = (self.xv[i]-x)/(self.xv[i+1]-x)
				elif outside(self.xv[i], self.yv[j]) and not outside(self.xv[i+1], self.yv[j]):
					x = pointOfIntersectionX(self.xv[i], self.xv[i+1], self.yv[j])
					self.tagsX[index] = index-2
					self.coeffsX[index] = (self.xv[i]-x)/(self.xv[i-1]-x)

				# tagsY
				if outside(self.xv[i], self.yv[j]) and not outside(self.xv[i], self.yv[j-1]):
					y = pointOfIntersectionY(self.yv[j-1], self.yv[j], self.xv[i])
					self.tagsY[index] = index+2*N
					self.coeffsY[index] = (self.yv[j]-y)/(self.yv[j+1]-y)
				elif outside(self.xv[i], self.yv[j]) and not outside(self.xv[i], self.yv[j+1]):
					y = pointOfIntersectionY(self.yv[j], self.yv[j+1], self.xv[i])
					self.tagsY[index] = index-2*N
					self.coeffsY[index] = (self.yv[j]-y)/(self.yv[j-1]-y)

				index+=1

		#print np.reshape(self.tagsX[::2], (N,N))
		#print np.reshape(self.tagsX[1::2], (N,N))
		#print np.reshape(self.tagsY[::2], (N,N))
		#print np.reshape(self.tagsY[1::2], (N,N))

	def tagInsidePoints(self):
		N = self.N
		h = self.h
		for j in xrange(1,N-1):
			for i in xrange(1,N-1):
				index = 2*(j*N+i)
				# tagsX
				if inside(self.xu[i], self.yu[j]) and not inside(self.xu[i-1], self.yu[j]):
					x = pointOfIntersectionX(self.xu[i-1], self.xu[i], self.yu[j])
					self.tagsX[index] = index+2
					self.coeffsX[index] = (self.xu[i]-x)/(self.xu[i+1]-x)
				elif inside(self.xu[i], self.yu[j]) and not inside(self.xu[i+1], self.yu[j]):
					x = pointOfIntersectionX(self.xu[i], self.xu[i+1], self.yu[j])
					self.tagsX[index] = index-2
					self.coeffsX[index] = (self.xu[i]-x)/(self.xu[i-1]-x)

				# tagsY
				if inside(self.xu[i], self.yu[j]) and not inside(self.xu[i], self.yu[j-1]):
					y = pointOfIntersectionY(self.yu[j-1], self.yu[j], self.xu[i])
					self.tagsY[index] = index+2*N
					self.coeffsY[index] = (self.yu[j]-y)/(self.yu[j+1]-y)
				elif inside(self.xu[i], self.yu[j]) and not inside(self.xu[i], self.yu[j+1]):
					y = pointOfIntersectionY(self.yu[j], self.yu[j+1], self.xu[i])
					self.tagsY[index] = index-2*N
					self.coeffsY[index] = (self.yu[j]-y)/(self.yu[j-1]-y)

				index+=1
				# tagsX
				if inside(self.xv[i], self.yv[j]) and not inside(self.xv[i-1], self.yv[j]):
					x = pointOfIntersectionX(self.xv[i-1], self.xv[i], self.yv[j])
					self.tagsX[index] = index+2
					self.coeffsX[index] = (self.xv[i]-x)/(self.xv[i+1]-x)
				elif inside(self.xv[i], self.yv[j]) and not inside(self.xv[i+1], self.yv[j]):
					x = pointOfIntersectionX(self.xv[i], self.xv[i+1], self.yv[j])
					self.tagsX[index] = index-2
					self.coeffsX[index] = (self.xv[i]-x)/(self.xv[i-1]-x)

				# tagsY
				if inside(self.xv[i], self.yv[j]) and not inside(self.xv[i], self.yv[j-1]):
					y = pointOfIntersectionY(self.yv[j-1], self.yv[j], self.xv[i])
					self.tagsY[index] = index+2*N
					self.coeffsY[index] = (self.yv[j]-y)/(self.yv[j+1]-y)
				elif inside(self.xv[i], self.yv[j]) and not inside(self.xv[i], self.yv[j+1]):
					y = pointOfIntersectionY(self.yv[j], self.yv[j+1], self.xv[i])
					self.tagsY[index] = index-2*N
					self.coeffsY[index] = (self.yv[j]-y)/(self.yv[j-1]-y)

		#print np.reshape(self.tagsX[::2], (N,N))
		#print np.reshape(self.tagsX[1::2], (N,N))
		#print np.reshape(self.tagsY[::2], (N,N))
		#print np.reshape(self.tagsY[1::2], (N,N))

	def plotTaggedPoints(self):
		N = self.N
		plt.ioff()
		fig = plt.figure()
		ax = fig.add_subplot(111)
		indices = [i for i,tagX in enumerate(self.tagsX) if tagX>-1]
		x = np.zeros(len(indices))
		y = np.zeros(len(indices))
		for I, index in enumerate(indices):
			idx = index/2
			i = idx % N
			j = idx / N
			if index % 2 == 0:
				x[I] = self.xu[i]
				y[I] = self.yu[j]
			else:
				x[I] = self.xv[i]
				y[I] = self.yv[j]
		ax.plot(x, y, 'ob')

		indices = [i for i,tagY in enumerate(self.tagsY) if tagY>-1]
		x = np.zeros(len(indices))
		y = np.zeros(len(indices))
		for I, index in enumerate(indices):
			idx = index/2
			i = idx % N
			j = idx / N
			if index % 2 == 0:
				x[I] = self.xu[i]
				y[I] = self.yu[j]
			else:
				x[I] = self.xv[i]
				y[I] = self.yv[j]
		ax.plot(x, y, 'xr', mew=1.5)

		ax.axis([0, 2*np.pi, 0, 2*np.pi])
		ax.grid(True)
		ax.set_xticks(np.linspace(0, 2*np.pi, N+1))
		ax.set_yticks(np.linspace(0, 2*np.pi, N+1))
		fig.gca().set_aspect('equal', adjustable='box')
		circ = plt.Circle((np.pi, np.pi), radius=np.pi/2., color='k', fill=False)
		ax.add_patch(circ)
		fig.savefig(self.folder+"/taggedPoints.png")
		fig.clf()

	def generateA(self):
		NavierStokesSolver.generateA(self)
		N = self.N
		index = 0
		for j in xrange(N):
			for i in xrange(N):
				if self.order == 'constant':
					# u
					if self.tagsX[index]>-1 or self.tagsY[index]>-1:
						start = self.A.indptr[index]
						self.A.data[start] = 0.
						self.A.data[start+1] = 0.
						self.A.data[start+2] = 1./self.dt
						self.A.data[start+3] = 0.
						self.A.data[start+4] = 0.
					index+=1

					# v
					if self.tagsX[index]>-1 or self.tagsY[index]>-1:
						start = self.A.indptr[index]
						self.A.data[start] = 0.
						self.A.data[start+1] = 0.
						self.A.data[start+2] = 1./self.dt
						self.A.data[start+3] = 0.
						self.A.data[start+4] = 0.
					index+=1

				if self.order == 'linear':
					# u
					start = self.A.indptr[index]
					if self.tagsX[index]>-1:
						self.A.data[start:start+5] = 0.
						self.A.data[start+2] = 1./self.dt
						for i,idx in enumerate(self.A.indices[start:start+5]):
							if idx==self.tagsX[index]:
								self.A.data[start+i] = -self.coeffsX[index]/self.dt
						#print "index:", index, "tagsX:", self.tagsX[index], "coeffsX:", self.coeffsX[index]
					elif self.tagsY[index]>-1:
						self.A.data[start:start+5] = 0.
						self.A.data[start+2] = 1./self.dt
						for i,idx in enumerate(self.A.indices[start:start+5]):
							if idx==self.tagsY[index]:
								self.A.data[start+i] = -self.coeffsY[index]/self.dt
					index+=1

					# v
					start = self.A.indptr[index]
					if self.tagsY[index]>-1:
						self.A.data[start:start+5] = 0.
						self.A.data[start+2] = 1./self.dt
						for i,idx in enumerate(self.A.indices[start:start+5]):
							if idx==self.tagsY[index]:
								self.A.data[start+i] = -self.coeffsY[index]/self.dt
					elif self.tagsX[index]>-1:
						self.A.data[start:start+5] = 0.
						self.A.data[start+2] = 1./self.dt
						for i,idx in enumerate(self.A.indices[start:start+5]):
							if idx==self.tagsX[index]:
								self.A.data[start+i] = -self.coeffsX[index]/self.dt
					index+=1
		#print self.A.data

	def calculateRN(self):
		NavierStokesSolver.calculateRN(self)
		N = self.N
		index = 0
		for j in xrange(N):
			for i in xrange(N):
				# u
				if self.tagsX[index]>-1 or self.tagsY[index]>-1:
					self.rn[index] = 0.
				index+=1

				# v
				if self.tagsX[index]>-1 or self.tagsY[index]>-1:
					self.rn[index] = 0.
				index+=1

	def zeroFluxesInsideBody(self):
		N = self.N
		halo = 1.0
		self.qZeroed[:] = self.q[:]
		index = 0
		for j in xrange(N):
			for i in xrange(N):
				# u
				if not outside(self.xu[i], self.yu[j], np.pi/2.*halo):
					self.qZeroed[index] = 0.
				index+=1

				# v
				if not outside(self.xv[i], self.yv[j], np.pi/2.*halo):
					self.qZeroed[index] = 0.
				index+=1

	def createMask(self):
		N = self.N
		halo = 1.0
		umask = np.ones(N*N)
		vmask = np.ones(N*N)
		index = 0
		for j in xrange(N):
			for i in xrange(N):
				# u
				if not outside(self.xu[i], self.yu[j], np.pi/2.*halo):
					umask[index] = 0.

				# v
				if not outside(self.xv[i], self.yv[j], np.pi/2.*halo):
					vmask[index] = 0.
				
				index+=1
		return np.reshape(umask, (N, N)), np.reshape(vmask, (N, N))

	def writeData(self, n):
		h = self.h
		N = self.N

		# u-velocity
		self.zeroFluxesInsideBody()
		U = np.zeros(N*N)
		U[:] = self.qZeroed[::2]/h
		#U[:] = self.q[::2]/h
		U = np.reshape(U, (N,N))
		x = np.linspace(h, 2*np.pi, N)
		y = np.linspace(0.5*h, 2*np.pi-0.5*h, N)
		X, Y = np.meshgrid(x, y)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		CS = ax.contour(X, Y, U, levels=np.linspace(-2., 2., 21))
		fig.colorbar(CS)
		ax.axis([0, 2*np.pi, 0, 2*np.pi])
		#ax.grid(True)
		#ax.set_xticks(np.linspace(0, 2*np.pi, self.coarsest+1))
		#ax.set_yticks(np.linspace(0, 2*np.pi, self.coarsest+1))
		fig.gca().set_aspect('equal', adjustable='box')
		circ = plt.Circle((np.pi, np.pi), radius=np.pi/2., color='k', fill=False)
		ax.add_patch(circ)
		fig.savefig("%s/u%07d.png" % (self.folder,n))
		fig.clf()

		# v-velocity
		V = np.zeros(N*N)
		V[:] = self.qZeroed[1::2]/h
		#V[:] = self.q[1::2]/h
		V = np.reshape(V, (N,N))
		x = np.linspace(0.5*h, 2*np.pi-0.5*h, N)
		y = np.linspace(h, 2*np.pi, N)
		X, Y = np.meshgrid(x, y)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		CS = ax.contour(X, Y, V, levels=np.linspace(-2., 2., 21))
		fig.colorbar(CS)
		ax.axis([0, 2*np.pi, 0, 2*np.pi])
		#ax.grid(True)
		#ax.set_xticks(np.linspace(0, 2*np.pi, self.coarsest+1))
		#ax.set_yticks(np.linspace(0, 2*np.pi, self.coarsest+1))
		fig.gca().set_aspect('equal', adjustable='box')
		circ = plt.Circle((np.pi, np.pi), radius=np.pi/2., color='k', fill=False)
		ax.add_patch(circ)
		fig.savefig("%s/v%07d.png" % (self.folder,n))
		fig.clf()

		# pressure

	def stepTime(self):
		NavierStokesSolver.stepTime(self)
		
		# mass conservation
		sum_fluxes = np.sum(self.QT * self.q)
		self.net_flux = np.append(self.net_flux, sum_fluxes)

	def runSimulation(self, nt=20, nsave=1, plot=True):
		NavierStokesSolver.runSimulation(self, nt=20, nsave=1, plot=True)

		if plot:
			plt.ioff()
			plt.clf()
			plt.plot(np.arange(1,nt+1), self.net_flux)
			#plt.axis([0., nt, -1., 1.])
			plt.savefig("%s/net_flux.png" % (self.folder))

if __name__ == "__main__":
	solver = FadlunEtAlSolver(N=80, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=1./np.pi, side='inside', folder="FadlunEtAl/flow-linear-inside")
	solver.runSimulation(nt=20, nsave=1)

	solver = FadlunEtAlSolver(N=80, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=1./np.pi, order='constant', side='inside', folder="FadlunEtAl/flow-constant-inside")
	solver.runSimulation(nt=20, nsave=1)

	solver = FadlunEtAlSolver(N=80, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=1./np.pi, folder="FadlunEtAl/flow-linear-outside")
	solver.runSimulation(nt=20, nsave=1)

	solver = FadlunEtAlSolver(N=80, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=1./np.pi, order='constant', folder="FadlunEtAl/flow-constant-outside")
	solver.runSimulation(nt=20, nsave=1)
