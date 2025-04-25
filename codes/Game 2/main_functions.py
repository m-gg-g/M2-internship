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
"""

oo =  np.inf

# direction to calculate delta Y and delta X from a position (i,j)
dirY = np.array([[-1, 1], [0, 0]])
dirX = np.array([[0, 0], [1, -1]])
# this one is to get the right value in the array of edges
dirYC = np.array([[0, 1], [0, 0]])
dirXC = np.array([[0, 0], [0, -1]])

class Data:
    def __init__(self, i=0, j=0, c=0):
        self.i = i
        self.j = j
        self.c = c

    def __lt__(self, other):
        return self.c < other.c

def rotate_array_counterclockwise(array):
    # transpose the array and reverse the rows of the transposed array
    rotated = np.flipud(array.T)
    return rotated

def solve(lv, Q, mat, dp, L, N):
    # Precompute i_vals and j_vals
    i_vals, j_vals = np.meshgrid(np.arange(1, L+1), np.arange(1, L+1), indexing='ij')
    
    if not lv:
        # create the mask for (i + j) & 1 condition
        condition = (i_vals + j_vals) % 2 == 1  # (i + j) is odd
        # assign values based on the condition
        dp[i_vals, j_vals, :] = np.where(condition[:, :, None], -oo, oo)
        Q[0].append(Data(N + 1, N + 1, 0))
        dp[N + 1][N + 1][0] = 0

    while Q[lv]:
        now = Q[lv].popleft()
        
        if dp[now.i, now.j, lv] != now.c:
            continue
        
        player = (now.i + now.j) & 1  # determine player based on position
        
        # direction updates
        ni = now.i + dirY[player]
        nj = now.j + dirX[player]
        nc = np.zeros_like(ni)  # Assuming nc is 0 for simplicity
        
        # create a mask for valid updates
        valid_mask = (ni >= 1) & (ni <= L) & (nj >= 1) & (nj <= L)
        
        # update dp values based on player
        if not player:
            update_mask = valid_mask & (dp[ni, nj, lv + 1] < nc)
            dp[ni[update_mask], nj[update_mask], lv + 1] = nc[update_mask]
            if lv < 2 * N:
                Q[lv + 1].extend([Data(ni[m], nj[m], nc[m]) for m in np.where(update_mask)[0]])
        else:
            update_mask = valid_mask & (dp[ni, nj, lv + 1] > nc)
            dp[ni[update_mask], nj[update_mask], lv + 1] = nc[update_mask]
            if lv < 2 * N:
                Q[lv + 1].extend([Data(ni[m], nj[m], nc[m]) for m in np.where(update_mask)[0]])

    if lv < 2 * N:
        solve(lv + 1, Q, mat, dp, L, N)
    else:
        return

    # vectorize the final dp updates
    player = (i_vals + j_vals) % 2
    
    # precompute ni, nj, and nc for both choices
    ni_0 = i_vals + dirY[player, 0]
    nj_0 = j_vals + dirX[player, 0]
    nc_0 = dp[ni_0, nj_0, lv + 1] + mat[player, i_vals + dirYC[player, 0], j_vals + dirXC[player, 0]]
    
    ni_1 = i_vals + dirY[player, 1]
    nj_1 = j_vals + dirX[player, 1]
    nc_1 = dp[ni_1, nj_1, lv + 1] + mat[player, i_vals + dirYC[player, 1], j_vals + dirXC[player, 1]]
    
    # determine the final dp values based on player
    dp[i_vals, j_vals, lv] = np.where((player == 0) & (nc_0 < nc_1), nc_1,
                                      np.where((player == 1) & (nc_0 > nc_1), nc_1, nc_0))

def main():
    global T, L, N

    T = 2
    while T > 0:
        T -= 1
        L, N = 3, 1
        MAXN = 2*L
        
        c = generate_costs(L, 0.5)

        mat = np.zeros((2, MAXN, MAXN))
        mat[0, 1:L+1, 1:L+1] = np.array(rotate_array_counterclockwise(c[1,:,:]))
        mat[1, 1:L+1, 1:L+1] = np.array(rotate_array_counterclockwise(c[0,:,:]))

        dp = np.zeros((MAXN, MAXN, MAXN))
        Q = [deque() for _ in range(MAXN)]

        solve(0, Q, mat, dp, L, N)
        print(dp[N + 1][N + 1][0])
