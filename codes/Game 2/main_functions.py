# import time 
import numpy as np
import matplotlib.pyplot as plt

"""
	p = occupation probability
	L = size of the lattice
	N : we play until the N-stage game.
"""

def generate_costs(L, p):
	"""
		Edges costs of the lattice according to a Bernoulli distribution with probability p.
		c[1, i, j] cost of the edge going upwards from (i, j), c[0] cost of the edge going rightward from (i, j).
	"""
	U = np.random.rand(2, L, L) 
	c = (U < p).astype(float) 
	return c

def calculate_optimal_strategy(first_steps_of_optimal_paths, n, UP, DOWN, UR, UL, DR, DL):
	"""
		For each z = (i, j) and k = [1..N], 
		calculates the best first move in the k-stage game starting from z.
	"""
	ur = np.argwhere((UP > DOWN) & (UR <= UL))
	first_steps_of_optimal_paths[n, ur[:, 0] + n, ur[:, 1] + n] = 0
	ul = np.argwhere((UP > DOWN) & (UR > UL))
	first_steps_of_optimal_paths[n, ul[:, 0] + n, ul[:, 1] + n] = 1
	dr = np.argwhere((UP <= DOWN) & (DR <= DL))
	first_steps_of_optimal_paths[n, dr[:, 0] + n, dr[:, 1] + n] = 2
	dl = np.argwhere((UP <= DOWN) & (DR > DL))
	first_steps_of_optimal_paths[n, dl[:, 0] + n, dl[:, 1] + n] = 3
	return first_steps_of_optimal_paths

def calculate_value(L, N, c):
	"""
		For each position (i, j) in Z^2,
		calculate the value of the largest possible n-stage game that can be played.
	"""
	value_matrix = np.zeros((L,L))

	for n in range(1, N + 1):
		UR = c[0, n:L-n, n+1:L-n+1] + value_matrix[n+1:L-n+1, n+1:L-n+1]
		UL = c[0, n-1:L-n-1, n+1:L-n+1] + value_matrix[n-1:L-n-1, n+1:L-n+1]
		player_2 = np.minimum(UR, UL) # player_2 decision
		UP = c[1, n:L-n, n:L-n] + player_2
		
		DR = c[0, n:L-n, n-1:L-n-1] + value_matrix[n+1:L-n+1, n-1:L-n-1]
		DL = c[0, n-1:L-n-1, n-1:L-n-1] + value_matrix[n-1:L-n-1, n-1:L-n-1]
		player_2 = np.minimum(DR, DL)	
		DOWN = c[1, n:L-n, n-1:L-n-1] + player_2 # player_2 decision

		value_matrix[n:L-n, n:L-n] = np.maximum(UP, DOWN) # player_1 decision
	return value_matrix

def calculate_value_and_optimal_strategy(L, N, c):
	"""
		For each position (i, j) in Z^2,
		calculate the first optimal move for every m-stage game for m in [1..N].
	"""
	value_matrix = np.zeros((L,L))
	first_steps_of_optimal_paths = np.array([[[0 for _ in range(L)] 
												 for _ in range(L)] 
												 for _ in range(N + 1)])
	for n in range(1, N + 1):
		UR = c[0, n:L-n, n+1:L-n+1] + value_matrix[n+1:L-n+1, n+1:L-n+1]
		UL = c[0, n-1:L-n-1, n+1:L-n+1] + value_matrix[n-1:L-n-1, n+1:L-n+1]
		player_2 = np.minimum(UR, UL) # player_2 decision
		UP = c[1, n:L-n, n:L-n] + player_2
		
		DR = c[0, n:L-n, n-1:L-n-1] + value_matrix[n+1:L-n+1, n-1:L-n-1]
		DL = c[0, n-1:L-n-1, n-1:L-n-1] + value_matrix[n-1:L-n-1, n-1:L-n-1]
		player_2 = np.minimum(DR, DL)	
		DOWN = c[1, n:L-n, n-1:L-n-1] + player_2 # player_2 decision

		value_matrix[n:L-n, n:L-n] = np.maximum(UP, DOWN) # player_1 decision
		first_steps_of_optimal_paths = calculate_optimal_strategy(first_steps_of_optimal_paths, n, UP, DOWN, UR, UL, DR, DL)
	return value_matrix, first_steps_of_optimal_paths

"""
	Other ways of implementing the computation of the n-stage gama value.
"""
def calculate_value_NN(L, N, c):
	value_matrix = np.zeros((L, L))
	value_matrix[1:L-1, :] = np.minimum(c[0, 1:L-1, :], c[0, 0:L-2, :])

	for stage in range(1, L):
		k = (stage + 1) // 2
		for x in range(k, L-k):
			for y in range(k, L-k):
				if (x+y)%2 != 0 and k%2 != 0 and k != 1:  # player 2's turn (minimizing)
					value_matrix[x, y] = min(
						c[0, x, y] + value_matrix[x+1, y],
						c[0, x-1, y] + value_matrix[x-1, y])
				if (x+y)%2 == 0 and k%2 == 0:  # player 1's turn (maximizing)
					value_matrix[x, y] = max(
						c[1, x, y] + value_matrix[x, y+1],
						c[1, x, y-1] + value_matrix[x, y-1])
	return value_matrix[N][N]

def calculate_value_NN_vectorized(L, N, c):
    value_matrix = np.zeros((L, L))
    value_matrix[1:L-1, :] = np.minimum(c[0, 1:L-1, :], c[0, 0:L-2, :])

    for stage in range(1, L):
        k = (stage + 1) // 2
        mask = np.zeros((L, L), dtype=bool)
        mask[k:L-k, k:L-k] = True

        player2_mask = mask & ((np.arange(L)[:, None] + np.arange(L)) % 2 != 0) & (k % 2 != 0) & (k != 1)
        player1_mask = mask & ((np.arange(L)[:, None] + np.arange(L)) % 2 == 0) & (k % 2 == 0)

        # Player 2's turn (minimizing)
        value_matrix[player2_mask] = np.minimum(
            c[0, player2_mask] + value_matrix[np.roll(player2_mask, -1, axis=0)],
            c[0, np.roll(player2_mask, 1, axis=0)] + value_matrix[np.roll(player2_mask, 1, axis=0)]
        )

        # Player 1's turn (maximizing)
        value_matrix[player1_mask] = np.maximum(
            c[1, player1_mask] + value_matrix[np.roll(player1_mask, -1, axis=1)],
            c[1, np.roll(player1_mask, 1, axis=1)] + value_matrix[np.roll(player1_mask, 1, axis=1)]
        )

    return value_matrix[N, N]

def dfs_calculate_value_NN(L, N, c):
	value_matrix = np.zeros((L, L))
	value_matrix[1:L-1, 0] = np.minimum(c[0, 1:L-1, 0], c[0, 0:L-2, 0])
	value_matrix[1:L-1, L-1] = np.minimum(c[0, 1:L-1, L-1], c[0, 0:L-2, L-1])

	def dfs(x, y, player_turn):
		if value_matrix[x, y] != 0:  # already calculated
			return value_matrix[x, y]

		if player_turn == 2:  # player 2's turn (minimizing)
			min_value = float('inf')
			# Try moving right (x+1, y) and left (x-1, y)
			if (x + 1) < L and (x - 1) >= 0:
				    min_value = min(min_value, c[0, x, y] + dfs(x + 1, y, 1))
				    min_value = min(min_value, c[0, x - 1, y] + dfs(x - 1, y, 1))
			value_matrix[x, y] = min_value
		else:  # player 1's turn (maximizing)
			max_value = float('-inf')
			# Try moving up (x, y+1) and down (x, y-1)
			if (y + 1) < L and (y - 1) >= 0: 
			    max_value = max(max_value, c[1, x, y] + dfs(x, y + 1, 2))
			    max_value = max(max_value, c[1, x, y - 1] + dfs(x, y - 1, 2))
			value_matrix[x, y] = max_value

		return value_matrix[x, y]
	return dfs(N, N, 1)  # Start with player 1

"""
	Under development!
	Not good.
"""
# from collections import deque

# def bfs_calculate_value_NN(L, N, c):
#     value_matrix = np.zeros((L, L))

#     value_matrix[1:L-1, 0] = np.minimum(c[0, 1:L-1, 0], c[0, 0:L-2, 0])
#     value_matrix[1:L-1, L-1] = np.minimum(c[0, 1:L-1, L-1], c[0, 0:L-2, L-1])

#     queue = deque([(N, N, 1)])  # (x, y, player_turn)
#     visited = set()

#     while queue:
#         x, y, player_turn = queue.popleft()

#         if (x, y) in visited:
#             continue
#         visited.add((x, y))

#         if value_matrix[x, y] != 0:  # already calculated
#             continue

#         if player_turn == 2:  # player 2's turn (minimizing)
#             min_value = float('inf')
#             if (x + 1) < L and (x - 1) >= 0:
#                 min_value = min(min_value, c[0, x, y] + value_matrix[x + 1, y])
#                 min_value = min(min_value, c[0, x - 1, y] + value_matrix[x - 1, y])
#                 queue.append((x + 1, y, 1))
#                 queue.append((x - 1, y, 1))
#             value_matrix[x, y] = min_value
#         else:  # player 1's turn (maximizing)
#             max_value = float('-inf')
#             if (y + 1) < L and (y - 1) >= 0:
#                 max_value = max(max_value, c[1, x, y] + value_matrix[x, y + 1])
#                 max_value = max(max_value, c[1, x, y - 1] + value_matrix[x, y - 1])
#                 queue.append((x, y + 1, 2))
#                 queue.append((x, y - 1, 2))
#             value_matrix[x, y] = max_value

#     return value_matrix[N, N]


"""
	To test the functions.
"""
# N_p = 5
# p = np.linspace(0.0, 1.0, num = N_p, endpoint = False)
# L_ = [3, 5, 7]

# if __name__ == "__main__":
# 	it = 0
# 	for L in L_:
# 		N = int(L/2) - (L % 2 == 0) # we consider games up to N-stage
# 		v_Np = np.zeros((len(L_), N_p)) 
		
# 		start = time.process_time()			
# 		for t in range(N_p):
# 			s = 0
# 			# if t%10==0: 
# 			# 	print("the time es:", t)
# 			# print(L, N)
# 			c = generate_costs(L, p[t])
# 			print(c)
# 			value_NN = calculate_value_NN(L, N, c)
# 			v_Np[it][t] = value_NN
# 		it+=1
# 	print(v_Np)