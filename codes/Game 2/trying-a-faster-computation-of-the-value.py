import numpy as np
from collections import deque

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
    # Step 1 and 2: Transpose the array and Reverse the rows of the transposed array
    rotated = np.flipud(array.T)
    return rotated

def solve(lv, Q, mat, dp, L, N):
    # Precompute i_vals and j_vals
    i_vals, j_vals = np.meshgrid(np.arange(1, L+1), np.arange(1, L+1), indexing='ij')
    
    if not lv:
        # Create the mask for (i + j) & 1 condition
        condition = (i_vals + j_vals) % 2 == 1  # (i + j) is odd
        # Assign values based on the condition
        dp[i_vals, j_vals, :] = np.where(condition[:, :, None], -oo, oo)
        Q[0].append(Data(N + 1, N + 1, 0))
        dp[N + 1][N + 1][0] = 0

    while Q[lv]:
        now = Q[lv].popleft()
        
        if dp[now.i, now.j, lv] != now.c:
            continue
        
        player = (now.i + now.j) & 1  # Determine player based on position
        
        # Vectorize the direction updates
        ni = now.i + dirY[player]
        nj = now.j + dirX[player]
        nc = np.zeros_like(ni)  # Assuming nc is 0 for simplicity
        
        # Create a mask for valid updates
        valid_mask = (ni >= 1) & (ni <= L) & (nj >= 1) & (nj <= L)
        
        # Update dp values based on player
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

    # Vectorize the final dp updates
    player = (i_vals + j_vals) % 2
    
    # Precompute ni, nj, and nc for both choices
    ni_0 = i_vals + dirY[player, 0]
    nj_0 = j_vals + dirX[player, 0]
    nc_0 = dp[ni_0, nj_0, lv + 1] + mat[player, i_vals + dirYC[player, 0], j_vals + dirXC[player, 0]]
    
    ni_1 = i_vals + dirY[player, 1]
    nj_1 = j_vals + dirX[player, 1]
    nc_1 = dp[ni_1, nj_1, lv + 1] + mat[player, i_vals + dirYC[player, 1], j_vals + dirXC[player, 1]]
    
    # Determine the final dp values based on player
    dp[i_vals, j_vals, lv] = np.where((player == 0) & (nc_0 < nc_1), nc_1,
                                      np.where((player == 1) & (nc_0 > nc_1), nc_1, nc_0))

"""
    Further improvement, on progress!!
"""

def solve(lv, Q, mat, dp, L, N):
    if not lv:
        # Create the mask for (i + j) & 1 condition
        i_vals, j_vals = np.meshgrid(np.arange(1, L+1), np.arange(1, L+1), indexing='ij')
        condition = (i_vals + j_vals) % 2 == 1  # (i + j) is odd
        # Assign values based on the condition
        dp[i_vals, j_vals, :] = np.where(condition[:, :, None], -oo, oo)
        Q[0].append(Data(N + 1, N + 1, 0))
        dp[N + 1][N + 1][0] = 0

     # while Q[lv]:
     #    now = Q[lv].popleft()
        
     #    # Skip if dp value has already been updated
     #    if dp[now.i, now.j, lv] != now.c:
     #        continue
        
     #    player = (now.i + now.j) & 1  # Determine player based on position
        
     #    # Vectorized direction updates
     #    ni = now.i + dirY[player]
     #    nj = now.j + dirX[player]
     #    nc = np.zeros_like(ni)  # Assuming nc is zero for simplicity
        
     #    # Mask for valid indices
     #    valid_mask = (ni >= 1) & (ni <= L) & (nj >= 1) & (nj <= L)
        
     #    # Update dp values based on player
     #    if not player:
     #        update_mask = valid_mask & (dp[ni, nj, lv + 1] < nc)
     #        dp[ni[update_mask], nj[update_mask], lv + 1] = nc[update_mask]
     #        if lv < 2 * N:
     #            Q[lv + 1].extend([Data(ni[m], nj[m], nc[m]) for m in np.where(update_mask)[0]])
     #    else:
     #        update_mask = valid_mask & (dp[ni, nj, lv + 1] > nc)
     #        dp[ni[update_mask], nj[update_mask], lv + 1] = nc[update_mask]
     #        if lv < 2 * N:
     #            Q[lv + 1].extend([Data(ni[m], nj[m], nc[m]) for m in np.where(update_mask)[0]])

    while Q[lv]:
        now = Q[lv].popleft()
        if dp[now.i][now.j][lv] != now.c:
            continue
        player = (now.i + now.j) & 1
        for l in range(2):
            ni = now.i + dirY[player][l]
            nj = now.j + dirX[player][l]
            nc = 0  # in this case player is the same as axis 0 or 1, Y or X
            nxt = Data(ni, nj, nc)
            # print(nxt.i, nxt.j, lv+1)
            if (not player and dp[nxt.i][nxt.j][lv + 1] < nxt.c) or (player and dp[nxt.i][nxt.j][lv + 1] > nxt.c):
                dp[nxt.i][nxt.j][lv + 1] = nxt.c
                if lv < 2 * N:
                    Q[lv + 1].append(nxt)

    if lv < 2 * N:
        solve(lv + 1, Q, mat, dp, L, N)
    else:
        return

    # # Vectorize final dp updates using np.where
    # player = (i_vals + j_vals) % 2
    
    # ni_0 = i_vals + dirY[player, 0]
    # nj_0 = j_vals + dirX[player, 0]
    # nc_0 = dp[ni_0, nj_0, lv + 1] + mat[player, i_vals + dirYC[player, 0], j_vals + dirXC[player, 0]]
    
    # ni_1 = i_vals + dirY[player, 1]
    # nj_1 = j_vals + dirX[player, 1]
    # nc_1 = dp[ni_1, nj_1, lv + 1] + mat[player, i_vals + dirYC[player, 1], j_vals + dirXC[player, 1]]
    
    # dp[i_vals, j_vals, lv] = np.where(
    #     (player == 0) & (nc_0 < nc_1), nc_1, np.where(
    #         (player == 1) & (nc_0 > nc_1), nc_1, nc_0
    #     )
    # )

    for i in range(1, L + 1):
        for j in range(1, L + 1):
            if not dp[i][j][lv]:
                player = (i + j) & 1
                choices = [None, None]
                for l in range(2):
                    ni = i + dirY[player][l]
                    nj = j + dirX[player][l]
                    nc = dp[ni][nj][lv + 1] + mat[player][i + dirYC[player][l]][j + dirXC[player][l]]
                    choices[l] = Data(ni, nj, nc)
                if (not player and choices[0].c < choices[1].c) or (player and choices[0].c > choices[1].c):
                    dp[i][j][lv] = choices[1].c
                else:
                    dp[i][j][lv] = choices[0].c

def main():
    global T, L, N

    T = 2
    while T > 0:
        T -= 1
        # Read L and N
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

if __name__ == "__main__":
    main()