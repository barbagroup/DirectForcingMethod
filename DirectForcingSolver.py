from NavierStokesSolver import NavierStokesSolver
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy.linalg as la
import matplotlib.pyplot as plt
import os

def outside(x, y):
	return (x-np.pi)**2 + (y-np.pi)**2 >= (np.pi/2.)**2

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

class DirectForcingSolver(NavierStokesSolver):
	def __init__(self, N=4, alphaImplicit=1., alphaExplicit=0., gamma=1., zeta=0., nu=0.01, dt=-1.0, order='linear'):
		NavierStokesSolver.__init__(self, N, alphaImplicit, alphaExplicit, gamma, zeta, nu, dt)
		self.order = order

	def initVecs(self):
		NavierStokesSolver.initVecs(self)
		N = self.N
		self.tagsX   = -np.ones(2*N*N, dtype=np.int)
		self.coeffsX = np.zeros(2*N*N)
		self.tagsY   = -np.ones(2*N*N, dtype=np.int)
		self.coeffsY = np.zeros(2*N*N)
		self.xu = -np.zeros(N)
		self.yu = -np.zeros(N)
		self.xv = -np.zeros(N)
		self.yv = -np.zeros(N)
		self.initCoords()
		self.tagPoints()

	def initMatrices(self):
		NavierStokesSolver.initMatrices(self)

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

	def tagPoints(self):
		self.tagOutsidePoints()

	def tagOutsidePoints(self):
		N = self.N
		h = self.h
		for j in xrange(1,N-1):
			for i in xrange(1,N-1):
				index = 2*(j*N+i)
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

		#print np.reshape(self.tagsX[::2], (N,N))
		#print np.reshape(self.tagsX[1::2], (N,N))
		#print np.reshape(self.tagsY[::2], (N,N))
		#print np.reshape(self.tagsY[1::2], (N,N))

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

	def generateBNQ(self):
		N = self.N
		rows = np.zeros(4*N*N, dtype=np.int)
		cols = np.zeros(4*N*N, dtype=np.int)
		vals = np.zeros(4*N*N, dtype=np.float)
		index = 0
		row_index = 0
		for j in xrange(N):
			for i in xrange(N):
				# u
				rows[index] = row_index
				cols[index] = j*N+i
				if self.tagsX[row_index]==-1 and self.tagsY[row_index]==-1:
					vals[index] = -1.
				else:
					vals[index] = 0.
				index+=1

				rows[index] = row_index
				cols[index] = j*N+i+1 if i<N-1 else j*N
				if self.tagsX[row_index]==-1 and self.tagsY[row_index]==-1:
					vals[index] = 1.
				else:
					vals[index] = 0.
				index+=1

				row_index+=1

				# v
				rows[index] = row_index
				cols[index] = j*N+i
				if self.tagsX[row_index]==-1 and self.tagsY[row_index]==-1:
					vals[index] = -1.
				else:
					vals[index] = 0.
				index+=1

				rows[index] = row_index
				cols[index] = (j+1)*N+i if j<N-1 else i
				if self.tagsX[row_index]==-1 and self.tagsY[row_index]==-1:
					vals[index] = 1.
				else:
					vals[index] = 0.
				index+=1

				row_index+=1

		self.BNQ = sp.csr_matrix((vals, (rows, cols)), shape=(2*N*N, N*N))
		self.QT = self.BNQ.transpose(copy=True)
		self.BNQ = self.dt*self.BNQ

if __name__ == "__main__":
	solver = DirectForcingSolver(N=80, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=0.1)
	solver.runSimulation(nt=20, nsave=1, folder="test-linear")

	solver = DirectForcingSolver(N=80, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=1./np.pi, order='constant')
	solver.runSimulation(nt=20, nsave=1, folder="test-constant")
