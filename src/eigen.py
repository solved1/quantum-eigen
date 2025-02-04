import numpy as np
from scipy.linalg import eigh
import argparse

def build_2d_hamiltonian(N=20, potential='well'):

	"""
	Build a discretized 2D Hamiltonian on an N x N grid

	Parameters
	----------
	N: 	int
		Number of points in each dimension (N^2 total points).
	potential: str
		Choose the potential. 'well' or 'harmonic' examples.

	Returns
	-------
	H: 	ndarray of shape (N^2, N^2)
		The Hamiltonian matrix approximating -d^2/dx^2 - d^2/dy^2 + V(x,y).
	 
	"""

	dx = 1. / float(N) #grid spacing, can be arbitraty
	inv_dx2 = float(N * N) #1/dx^2
	H = np.zeros((N*N, N*N), dtype=np.float64)

	#Helper function to map (i, j) -> linear index
	def idx(i, j):
		return i * N + j

	#Potential function
	def V(i, j):
		#Example 1: infinite square well --> zero in interioir, large outside
		if potential == 'well':
		#No boundary enforcement here, but can skip boundary wavefunction
			return 0.
		#Example 2: 2D harmonic oscillator around center
		elif potential == 'harmonic':
			x = (i - N/2) * dx
			y = (j - N/2) * dx
			#Quadratic potential V = k * (x^2 + y^2)
			return 4. * (x**2 + y**2)
		else:
			return 0.


	#Build the matrix: For exach (i, j), set diagonal for 2D Laplacial plus V
	for i in range(N):
		for j in range(N):
			row = idx(i, j)
			#Potential
			H[row, row] = -4. * inv_dx2 + V(i, j) # "Kinetic" ~ -4/dx^2 in 2D FD
			# Neighbors (assuming no boundary conditions or Dirichlet)
			if i > 0: #up
				H[row, idx(i-1, j)] = inv_dx2
			if i < N-1: #down
				H[row, idx(i+1, j)] = inv_dx2
			if j > 0: #left
				H[row, idx(i, j-1)] = inv_dx2
			if j < N-1: #right
				H[row, idx(i, j+1)] = inv_dx2

		return H

def solve_eigen(N=20, potential='well', n_eigs=None):
	"""
	Build a 2D Hamiltonian and solve for the lowest n_eigs eigenvalues.
	
	Parameters
	----------

	N :	int
		Grid points in each dimension.
	potential : str
		Potential type.
	n_eigs : int
		Number of eigenvalues to return.

	Returns
	-------
	
	vals :	array_like
		The lowest n_eigs eigenvalues sorted ascending.
	vecs :	array_like
		The corresponding eigenvectors.
	"""

	H = build_2d_hamiltonian(N, potential)
	#Solve entire spectrum (careful for large N)
	vals, vecs = eigh(H)
	#Sort
	idx_sorted = np.argsort(vals)
	vals_sorted = vals[idx_sorted]
	vecs_sorted = vecs[:, idx_sorted]
	if n_eigs is None:
		return vals_sorted, vecs_sorted
	else:
		return vals_sorted[:n_eigs], vecs_sorted[:, :n_eigs]

if __name__ == '__main__':
	
	def positive_int(value):
		"""Custom type checker for positive integers."""
		ivalue = int(value)
		if ivalue <= 0:
			raise argparse.ArgumentTypeError(f"{value} is an invalid positive integer value")
		return ivalue
	def smaller_than(value_1, value_2):
		"""Checks that the number of eigenvalues is equal or lesser than N^2."""
		if positive_int(value_1) > positive_int(value_2):
			raise Exception(f"{value_1} is an invalid number of eigenstates for N="+str(value_2))
		return 0
	
	#Example local test
	#vals, vecs = solve_eigen(N=10, potential='well', n_eigs=5)
	#print("Lowest 5 eigenvalues:", vals)

	parser = argparse.ArgumentParser()
	parser.add_argument('N', action='store', type=positive_int)
	parser.add_argument('potential', action='store',type=str)
	parser.add_argument('n_eigs', action='store', type=positive_int)
	args = parser.parse_args()
	#print("args", args)
	smaller_than(args.n_eigs, args.N**2)
	vals, vecs = solve_eigen(args.N, args.potential, args.n_eigs)
	print("Lowest ", args.n_eigs, " eigenvalues:", vals)
