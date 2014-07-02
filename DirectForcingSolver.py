from NavierStokesSolver import NavierStokesSolver
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy.linalg as la
import matplotlib.pyplot as plt
import os

def outside(x, y):
	return (x-np.pi)**2 + (y-np.pi)**2 > (np.pi/2.)**2

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
		N = self.N
		h = self.h
		for j in xrange(1,N-1):
			for i in xrange(1,N-1):
				index = 2*(j*N+i)
				if outside(self.xu[i], self.yu[j]) and not outside(self.xu[i-1], self.yu[j]):
					self.tagsX[index] = index+2 # change this for a different interpolation scheme
					x = pointOfIntersectionX(self.xu[i-1], self.xu[i], self.yu[j])
					self.coeffsX[index] = (self.xu[i]-x)/(self.xu[i-1]-x)
				elif outside(self.xu[i], self.yu[j]) and not outside(self.xu[i+1], self.yu[j]):
					self.tagsX[index] = index-2
					x = pointOfIntersectionX(self.xu[i], self.xu[i+1], self.yu[j])
					self.coeffsX[index] = (self.xu[i]-x)/(self.xu[i+1]-x)

				if outside(self.xu[i], self.yu[j]) and not outside(self.xu[i], self.yu[j-1]):
					self.tagsY[index] = index+2*N
					y = pointOfIntersectionY(self.yu[j-1], self.yu[j], self.xu[i])
					self.coeffsY[index] = (self.yu[j]-y)/(self.yu[j-1]-y)
				elif outside(self.xu[i], self.yu[j]) and not outside(self.xu[i], self.yu[j+1]):
					self.tagsY[index] = index-2*N
					y = pointOfIntersectionY(self.yu[j], self.yu[j+1], self.xu[i])
					self.coeffsY[index] = (self.yu[j]-y)/(self.yu[j+1]-y)

				index+=1
				if outside(self.xv[i], self.yv[j]) and not outside(self.xv[i-1], self.yv[j]):
					self.tagsX[index] = index+2
					x = pointOfIntersectionX(self.xv[i-1], self.xv[i], self.yv[j])
					self.coeffsX[index] = (self.xv[i]-x)/(self.xv[i-1]-x)
				elif outside(self.xv[i], self.yv[j]) and not outside(self.xv[i+1], self.yv[j]):
					self.tagsX[index] = index-2
					x = pointOfIntersectionX(self.xv[i], self.xv[i+1], self.yv[j])
					self.coeffsX[index] = (self.xv[i]-x)/(self.xv[i-1]-x)

				if outside(self.xv[i], self.yv[j]) and not outside(self.xv[i], self.yv[j-1]):
					self.tagsY[index] = index+2*N
					y = pointOfIntersectionY(self.yv[j-1], self.yv[j], self.xv[i])
					self.coeffsY[index] = (self.yv[j]-y)/(self.yv[j-1]-y)
				elif outside(self.xv[i], self.yv[j]) and not outside(self.xv[i], self.yv[j+1]):
					self.tagsY[index] = index-2*N
					y = pointOfIntersectionY(self.yv[j], self.yv[j+1], self.xv[i])
					self.coeffsY[index] = (self.yv[j]-y)/(self.yv[j+1]-y)
		print np.reshape(self.tagsX[::2], (N,N))
		print np.reshape(self.tagsX[1::2], (N,N))
		print np.reshape(self.tagsY[::2], (N,N))
		print np.reshape(self.tagsY[1::2], (N,N))

	def generateA(self):
		N = self.N
		NavierStokesSolver.generateA(self)
		print len(self.A.indices), 2*N*N
		print len(self.A.indptr), 2*N*N
		print self.A.indptr

if __name__ == "__main__":
	solver = DirectForcingSolver(N=5, alphaExplicit=0., alphaImplicit=1., nu=0.1)
	solver.initVecs()
	solver.initMatrices()