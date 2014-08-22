import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
import pyamg
import errno

class NavierStokesSolver:
	def __init__(self, N=4, alphaImplicit=1., alphaExplicit=0., gamma=1., zeta=0., nu=0.1, dt=-1.0, folder="."):
		self.nu = nu
		self.N = N
		self.h = 2*np.pi/N
		self.dt = self.h*self.h/self.nu/10.
		if dt>0:
			self.dt = dt
		print "nu =", self.nu
		print "h  =", self.h
		print "dt =", self.dt
		self.alphaExplicit = alphaExplicit
		self.alphaImplicit = alphaImplicit
		self.gamma = gamma
		self.zeta = zeta
		self.folder = folder

	def initVecs(self):
		N = self.N
		self.phi = np.zeros(N*N)
		self.q = np.zeros(2*N*N)
		self.qStar = np.zeros(2*N*N)
		self.rn = np.zeros(2*N*N)
		self.H = np.zeros(2*N*N)
		self.initFluxes()

	def initFluxes(self):
		h = self.h
		N = self.N
		index = 0
		for j in xrange(N):
			for i in xrange(N):
				# u
				x = (i+1)*h
				y = (j+0.5)*h
				self.q[index] = np.sin(x)*np.cos(y)*h
				index+=1

				# v
				x = (i+0.5)*h
				y = (j+1)*h
				self.q[index] = -np.cos(x)*np.sin(y)*h
				index+=1

	def exactSolutionTaylorGreen(self, t):
		h = self.h
		N = self.N
		self.exactSolution = np.zeros(2*N*N)
		index = 0
		for j in xrange(N):
			for i in xrange(N):
				# u
				x = (i+1)*h
				y = (j+0.5)*h
				self.exactSolution[index] = np.exp(-2*self.nu*t)*np.sin(x)*np.cos(y)*h
				index+=1

				# v
				x = (i+0.5)*h
				y = (j+1)*h
				self.exactSolution[index] = -np.exp(-2*self.nu*t)*np.cos(x)*np.sin(y)*h
				index+=1

	def initMatrices(self):
		self.generateA()
		self.generateBNQ()
		self.generateQTBNQ()

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
				vals[index] = -1.
				index+=1

				rows[index] = row_index
				cols[index] = j*N+i+1 if i<N-1 else j*N
				vals[index] = 1.
				index+=1

				row_index+=1

				# v
				rows[index] = row_index
				cols[index] = j*N+i
				vals[index] = -1.
				index+=1

				rows[index] = row_index
				cols[index] = (j+1)*N+i if j<N-1 else i
				vals[index] = 1.
				index+=1

				row_index+=1

		self.BNQ = sp.csr_matrix((vals, (rows, cols)), shape=(2*N*N, N*N))
		self.QT = self.BNQ.transpose(copy=True)
		self.BNQ = self.dt*self.BNQ
	
	def generateQTBNQ(self):
		self.QTBNQ = self.QT*self.BNQ
		idx = list(self.QTBNQ.indices).index(0)
		self.QTBNQ.data[idx] *=2.0
		self.ml = pyamg.ruge_stuben_solver(self.QTBNQ)

	def generateA(self):
		h = self.h
		dt = self.dt
		N = self.N
		alpha = self.alphaImplicit
		nu = self.nu

		rows = np.zeros(2*5*N*N, dtype=np.int)
		cols = np.zeros(2*5*N*N, dtype=np.int)
		vals = np.zeros(2*5*N*N, dtype=np.float)
		row_index = 0
		index = 0
		for j in xrange(N):
			for i in xrange(N):
				# u
				rows[index] = row_index
				cols[index] = 2*((j-1)*N+i) if j>0 else 2*((N-1)*N+i)
				vals[index] = -alpha*nu*1./h**2
				index+=1

				rows[index] = row_index
				cols[index] = 2*(j*N+(i-1)) if i>0 else 2*(j*N+(N-1))
				vals[index] = -alpha*nu*1./h**2
				index+=1

				rows[index] = row_index
				cols[index] = 2*(j*N+i)
				vals[index] = 1./dt + alpha*nu*4./h**2
				index+=1

				rows[index] = row_index
				cols[index] = 2*(j*N+(i+1)) if i<N-1 else 2*(j*N+0)
				vals[index] = -alpha*nu*1./h**2
				index+=1

				rows[index] = row_index
				cols[index] = 2*((j+1)*N+i) if j<N-1 else 2*(0*N+i)
				vals[index] = -alpha*nu*1./h**2
				index+=1

				row_index+=1
				
				# v
				rows[index] = row_index
				cols[index] = 2*((j-1)*N+i)+1 if j>0 else 2*((N-1)*N+i)+1
				vals[index] = -alpha*nu*1./h**2
				index+=1

				rows[index] = row_index
				cols[index] = 2*(j*N+(i-1))+1 if i>0 else 2*(j*N+(N-1))+1
				vals[index] = -alpha*nu*1./h**2
				index+=1

				rows[index] = row_index
				cols[index] = 2*(j*N+i)+1
				vals[index] = 1./dt + alpha*nu*4./h**2
				index+=1

				rows[index] = row_index
				cols[index] = 2*(j*N+(i+1))+1 if i<N-1 else 2*(j*N+0)+1
				vals[index] = -alpha*nu*1./h**2
				index+=1

				rows[index] = row_index
				cols[index] = 2*((j+1)*N+i)+1 if j<N-1 else 2*(0*N+i)+1
				vals[index] = -alpha*nu*1./h**2
				index+=1

				row_index+=1

		self.A = sp.csr_matrix((vals, (rows, cols)), shape=(2*N*N, 2*N*N))

	def calculateRN(self):
		alpha = self.alphaExplicit
		N = self.N
		h = self.h

		index = 0
		for j in xrange(N):
			for i in xrange(N):
				# u
				center = index
				south  = (index-2*N)%(2*N*N)
				west   = index-2 if i>0 else index+2*(N-1)
				east   = index+2 if i<N-1 else index-2*(N-1)
				north  = (index+2*N)%(2*N*N)

				diffusion_term = alpha*self.nu*(1./h**2)*(self.q[north] + self.q[south] + self.q[east] + self.q[west] - 4.*self.q[center])/h

				southwest = south+1
				southeast = southwest+2 if i<N-1 else southwest-2*(N-1)
				northwest = center+1
				northeast = northwest+2 if i<N-1 else northwest-2*(N-1)

				convection_term = (0.25/h**2)*((self.q[center]+self.q[east])*(self.q[center]+self.q[east]) - (self.q[center]+self.q[west])*(self.q[center]+self.q[west]))/h \
				                  + (0.25/h**2)*((self.q[center]+self.q[north])*(self.q[northwest]+self.q[northeast]) - (self.q[center]+self.q[south])*(self.q[southwest]+self.q[southeast]))/h

				self.H[index] = self.gamma*convection_term + self.zeta*self.H[index]
				self.rn[index] = (-self.H[index] + diffusion_term + self.q[index]/h/self.dt)*h

				index+=1

				# v
				center+= 1
				south += 1
				west  += 1
				east  += 1
				north += 1

				diffusion_term = alpha*self.nu*(1./h**2)*(self.q[north] + self.q[south] + self.q[east] + self.q[west] - 4.*self.q[center])/h

				northeast = north-1
				northwest = northeast-2 if i>0 else northeast+2*(N-1)
				southeast = center-1
				southwest = southeast-2 if i>0 else southeast+2*(N-1)

				convection_term = (0.25/h**2)*((self.q[center]+self.q[east])*(self.q[northeast]+self.q[southeast]) - (self.q[center]+self.q[west])*(self.q[northwest]+self.q[southwest]))/h \
				                  + (0.25/h**2)*((self.q[center]+self.q[north])*(self.q[center]+self.q[north]) - (self.q[center]+self.q[south])*(self.q[center]+self.q[south]))/h

				self.H[index] = self.gamma*convection_term + self.zeta*self.H[index]
				self.rn[index] = (-self.H[index] + diffusion_term + self.q[index]/h/self.dt)*h

				index+=1

	def stepTime(self):
		# solve for intermediate velocity
		self.calculateRN()
		self.qStar, _ = sla.bicgstab(self.A, self.rn, tol=1e-8)

		# solve for pressure
		self.rhs2 = self.QT*self.qStar
		#self.phi, _ = sla.cg(self.QTBNQ, self.rhs2)
		self.phi = self.ml.solve(self.rhs2, tol=1e-8)

		# projection step
		self.q = self.qStar - self.BNQ*self.phi

	def writeData(self, n):
		h = self.h
		N = self.N


		U = np.zeros(N*N)
		U[:] = self.q[::2]/h
		U = np.reshape(U, (N,N))
		x = np.linspace(h, 2*np.pi, N)
		y = np.linspace(0.5*h, 2*np.pi-0.5*h, N)
		X, Y = np.meshgrid(x, y)

		CS = plt.contour(X, Y, U, levels=np.linspace(-1., 1., 11))
		plt.colorbar(CS)
		plt.axis([0, 2*np.pi, 0, 2*np.pi])
		plt.gca().set_aspect('equal', adjustable='box')
		plt.savefig("%s/u%07d.png" % (self.folder,n))
		plt.clf()

		V = np.zeros(N*N)
		V[:] = self.q[1::2]/h
		V = np.reshape(V, (N,N))
		x = np.linspace(0.5*h, 2*np.pi-0.5*h, N)
		y = np.linspace(h, 2*np.pi, N)
		X, Y = np.meshgrid(x, y)

		CS = plt.contour(X, Y, V, levels=np.linspace(-1., 1., 11))
		plt.colorbar(CS)
		plt.axis([0, 2*np.pi, 0, 2*np.pi])
		plt.gca().set_aspect('equal', adjustable='box')
		plt.savefig("%s/v%07d.png" % (self.folder, n))
		plt.clf()

	def runSimulation(self, nt=20, nsave=1, plot=True):
		try:
			os.makedirs(self.folder)
		except OSError as exc:
			if exc.errno == errno.EEXIST and os.path.isdir(self.folder):
				pass
			else:
				raise
		self.initVecs()
		self.initMatrices()
		if plot:
			self.writeData(0)
		for n in xrange(1,nt+1):
			self.stepTime()
			if n%nsave==0 and plot:
				self.writeData(n)

if __name__ == "__main__":
	NT = 20

	solver = NavierStokesSolver(N=6, alphaExplicit=0., alphaImplicit=1., dt=0.1, folder="NS06")
	solver.runSimulation(nt=NT)

	solver = NavierStokesSolver(N=12, alphaExplicit=0., alphaImplicit=1., dt=0.1, folder="NS12")
	solver.runSimulation(nt=NT)

	solver = NavierStokesSolver(N=24, alphaExplicit=0., alphaImplicit=1., dt=0.1, folder="NS24")
	solver.runSimulation(nt=NT)