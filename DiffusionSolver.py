from DirectForcingSolver import DirectForcingSolver
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import numpy.linalg as la
import matplotlib.pyplot as plt
import os

class DiffusionSolver(DirectForcingSolver):
	def __init__(self, N=4, alphaImplicit=1., alphaExplicit=0., gamma=0., zeta=0., nu=0.01, dt=-1.0, folder=".", order='linear', side='outside'):
		DirectForcingSolver.__init__(self, N, alphaImplicit, alphaExplicit, gamma, zeta, nu, dt, folder)
		self.gamma = 0.
		self.zeta = 0.
		self.order = order
		self.side = side

	def stepTime(self):
		# solve for intermediate velocity
		self.calculateRN()
		self.qStar, _ = sla.cg(self.A, self.rn)

		# projection step
		self.q = self.qStar

if __name__ == "__main__":
	solver = DiffusionSolver(N=80, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=1./np.pi, folder="diffusion-linear-outside")
	solver.runSimulation(nt=20, nsave=1)

	solver = DiffusionSolver(N=80, alphaExplicit=0., alphaImplicit=1., nu=0.1, dt=1./np.pi, order='constant', folder="diffusion-constant-outside")
	solver.runSimulation(nt=20, nsave=1)