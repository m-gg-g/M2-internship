import numpy as np
import matplotlib.pyplot as plt

"""
	p = occupation probability.
	L = size of the lattice.
	N = we play until the N-stage game.
"""

def generate_costs(L, p):
	"""
		Edges costs of the lattice according to a Bernoulli distribution with probability p.
		c_aux[0] costs of the principal diagonals, c_aux[1] cost of the antidiagonals.
		c[k, i, j] cost of the diagonal in direction k from (i, j)
	"""
	U = np.random.rand(2, L, L) 
	c_aux = (U < p).astype(float) 

	c = np.array([[[0.0 for _ in range (L)] for _ in range (L)] for _ in range(4)])
	# c[k, i, j] = cost of the diagonal in the k-th direction starting from i,j
	c[1, 1:, :(L-1)] = c[2, :(L-1), 1: ] = c_aux[1, 1:, :(L-1)]
	c[0, :(L-1), :(L-1)] = c[3, 1:, 1:] = c_aux[0, :(L-1), :(L-1)]
	return c

def calculate_optimal_strategy(first_steps_of_optimal_paths, n, UP, DOWN, UR, UL, DR, DL):
	"""
		For each z = (i, j) and k = 1,..., N,
		calculate the best first move in the k-stage game starting from (i, j). 
	"""
	# saving the best first move for n-stage game
	ur = np.argwhere((UP < DOWN) & (UR >= UL))
	first_steps_of_optimal_paths[n, ur[:, 0] + n, ur[:, 1] + n] = 0
	ul = np.argwhere((UP < DOWN) & (UR < UL))
	first_steps_of_optimal_paths[n, ul[:, 0] + n, ul[:, 1] + n] = 1
	dr = np.argwhere((UP >= DOWN) & (DR >= DL))
	first_steps_of_optimal_paths[n, dr[:, 0] + n, dr[:, 1] + n] = 2
	dl = np.argwhere((UP >= DOWN) & (DR < DL))
	first_steps_of_optimal_paths[n, dl[:, 0] + n, dl[:, 1] + n] = 3
	return first_steps_of_optimal_paths

def calculate_value(L, N, c):
	"""
		For each position (i, j) in Z^2,
		calculate the value of the largest possible n-stage game that can be played.
	"""
	value_matrix = np.zeros((L,L))
	for n in range(1, N + 1):
		# if n % 20 == 0:
		# 	print(n)
		UR = c[0, n:L-n, n:L-n] + value_matrix[n+1:L-n+1, n+1:L-n+1]
		UL = c[1, n:L-n, n:L-n] + value_matrix[n-1:L-n-1, n+1:L-n+1]
		UP = np.maximum(UR, UL)

		DR = c[2, n:L-n, n:L-n] + value_matrix[n+1:L-n+1, n-1:L-n-1]
		DL = c[3, n:L-n, n:L-n] + value_matrix[n-1:L-n-1, n-1:L-n-1]
		DOWN = np.maximum(DR, DL)
		
		value_matrix[n:(L-n), n:(L-n)] = np.minimum(UP, DOWN)
	return value_matrix

def calculate_value_and_optimal_strategy(L, N, c):
	"""
		For each position (i, j) in Z^2,
		calculate the value of the largest possible n-stage game that can be played.
	"""
	value_matrix = np.zeros((L,L))
	first_steps_of_optimal_paths = np.array([[[0 for _ in range(L)] 
												 for _ in range(L)]
												 for _ in range(N+1)]) 
	for n in range(1, N + 1):
		# if n % 20 == 0:
		# 	print(n)
		UR = c[0, n:L-n, n:L-n] + value_matrix[n+1:L-n+1, n+1:L-n+1]
		UR = c[0, n:L-n, n:L-n] + value_matrix[n+1:L-n+1, n+1:L-n+1]
		UL = c[1, n:L-n, n:L-n] + value_matrix[n-1:L-n-1, n+1:L-n+1]
		UP = np.maximum(UR, UL)

		DR = c[2, n:L-n, n:L-n] + value_matrix[n+1:L-n+1, n-1:L-n-1]
		DL = c[3, n:L-n, n:L-n] + value_matrix[n-1:L-n-1, n-1:L-n-1]
		DOWN = np.maximum(DR, DL)
		
		value_matrix[n:(L-n), n:(L-n)] = np.minimum(UP, DOWN)
		first_steps_of_optimal_paths = calculate_optimal_strategy(first_steps_of_optimal_paths, n, UP, DOWN, UR, UL, DR, DL)
	return value_matrix, first_steps_of_optimal_paths
