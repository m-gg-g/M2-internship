import time 
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from main_functions import generate_costs, calculate_value

MAXITER = 30
N_p = 200
EPS = 0.01
true_values = [0.0, 0.25, 0.5, 0.75, 1.0]
p = np.linspace(0.0, 1., num = N_p, endpoint = False)

# L_ = np.linspace(81, 101, 1, dtype = int)
L_ = [1500]

def determine_critical_probabilities(p, N, v_Np):	
	# print(p, N, v_Np)
	t_c_lb = np.zeros(5).astype(int)
	t_c_up = np.zeros(5).astype(int)
	for k in range(5):	
		t_c_lb[k] = np.argwhere(abs(v_Np - true_values[k]) <= EPS)[0, 0]
		t_c_up[k] = np.argwhere(abs(v_Np - true_values[k]) <= EPS)[-1, 0]
	return t_c_lb, t_c_up

def determine_critical_probabilities_and_plot(p, N, v_Np):
		plt.figure(figsize=(8, 6))  # Adjust the figure size (wide enough for the legend)
		plt.plot(p, v_Np, zorder = 0)

		t_c_lb, t_c_up = determine_critical_probabilities(p, N, v_Np)
		for k in range(5):
			p_c_lb, p_c_up = p[t_c_lb[k]], p[t_c_up[k]]
			v_c_lb, v_c_up = v_Np[t_c_lb[k]], v_Np[t_c_up[k]]
			p_c_m, v_c_m = (p_c_lb + p_c_up)*0.5, (v_c_lb + v_c_up)*0.5 

			plt.scatter(round(p_c_lb, 2), v_c_lb, label = "r_" + str(int(true_values[k]) if k%5==0 else true_values[k]) + " = " + str(round(p_c_lb, 2)), color = "red", zorder = 1)
			plt.scatter(round(p_c_m, 2), v_c_m, label = "rm_" + str(int(true_values[k]) if k%5==0 else true_values[k]) + " = " + str(round(p_c_m, 2)), color = "black", zorder = 1)
			plt.scatter(round(p_c_up, 2), v_c_up, label = "r'_" + str(int(true_values[k]) if k%5==0 else true_values[k]) + " = " + str(round(p_c_up, 2)), color = "green", zorder = 1)
			
		plt.xlabel('p', fontweight='bold')
		plt.ylabel('v_' + str(N), fontweight = 'bold')
		plt.legend()

		# Add the legend outside the plot (to the right)
		plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
		# Adjust layout to make room for the legend
		plt.tight_layout()

		# plt.show()
		# plt.savefig('P1' + 'p' + "N" + str(N) + 'MAXITER' + str(MAXITER) + '.png')
		plt.savefig('centers' + 'p' + "N" + str(N) + 'MAXITER' + str(MAXITER) + '.eps',  format = 'eps', dpi=1200, bbox_inches='tight', pad_inches=0.1)
		plt.close()

if __name__ == "__main__":
	# print("slope in the center is:")
	for L in L_:
		# MAXN = 2*L
		N = int(L/2) - (L % 2 == 0) # we consider games up to N-stage
		v_Np = np.empty(N_p) # initializing the array for saving the value as a function of p

		start = time.process_time()			
		for t in range(N_p):
			import time
			s = 0
			if t%10==0: 
				print("Iteration: ", t)
			for it in range(MAXITER):
				c = generate_costs(L, p[t])

				value_matrix = calculate_value(L, N, c)
				value_NN = value_matrix[N][N]

				"""
					other ways for calculating the value
				"""
				
				# mat = np.zeros((2, MAXN, MAXN))
				# mat[0, 1:L+1, 1:L+1] = np.array(rotate_array_counterclockwise(c[1,:,:]))
				# mat[1, 1:L+1, 1:L+1] = np.array(rotate_array_counterclockwise(c[0,:,:]))
				# dp = np.zeros((MAXN, MAXN, MAXN))
				# Q = [deque() for _ in range(MAXN)]					
				# solve(0, Q, mat, dp, L, N)
				# value_NN = dp[N + 1][N + 1][0]

				# value_NN = calculate_value_NN_vectorized(L, N, c)

				s = s + value_NN/(2*N)
			v_Np[t] = s/MAXITER
			# if (t == 0) or (t == N_p/2): print('time is:', time.process_time() - start)


		# saving the arrays p, v_Np
		np.savetxt('pfor' + 'N' + str(N) + 'MAXITER' + str(MAXITER), p, delimiter=',')
		np.savetxt('v_Npfor' + 'N' + str(N) + 'MAXITER' + str(MAXITER), v_Np, delimiter='t')
		t_c_lb, t_c_up = determine_critical_probabilities_and_plot(p, N, v_Np)

		# p_1, p_2 = p[t_c_lb[2]], p[t_c_up[2]]
		# v_1, v_2 = v_Np[t_c_lb[2]], v_Np[t_c_up[2]]
		# slope_midpoint = (v_2-v_1)/(p_2 - p_1) 
		# print(slope_midpoint)
		