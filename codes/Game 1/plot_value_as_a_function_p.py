import time 
import numpy as np 
import matplotlib.pyplot as plt
from main_functions import generate_costs, calculate_value

MAXITER = 10
N_p = 200
# MAXITER_L = 5
EPS = 0.01
L_ = [2001]
p = np.linspace(0, 1, num = N_p, endpoint = False)

moves_x_diagonal = np.array([1, -1, 1, -1])
moves_y_diagonal = np.array([1, 1, -1, -1])

def determine_critical_probabilities(p, N, v_Np):
	# saving the arrays p, v_Np for future graphs
	np.savetxt('pfor' + 'N' + str(N) + 'MAXITER' + str(MAXITER), p, delimiter=',')
	np.savetxt('v_Npfor' + 'N' + str(N) + 'MAXITER' + str(MAXITER), v_Np, delimiter='t')

	t_c1 = np.argwhere(abs(v_Np - 1.0) <= EPS)[0, 0]
	t_c0 = np.argwhere(abs(v_Np - 0.0) <= EPS)[-1, 0]	
	return t_c0, t_c1

def determine_critical_probabilities_and_plot(p, N, v_Np):
	# t_p = (np.array([0.1, 0.5, 0.9])/0.005).astype(int)
	print("Para N = ", N)

	t_c0, t_c1 = determine_critical_probabilities(p, N, v_Np)
	p_c0, p_c1 = p[t_c0], p[t_c1] 
	v_c0, v_c1 = v_Np[t_c0], v_Np[t_c1]
	print(p_c0, v_c0)
	print(p_c1, v_c1)
	
	plt.plot(p, v_Np, zorder = 0)
	plt.scatter(round(p_c1, 5), v_c1, label = "p1 = " + str(p_c1), color = "red", zorder = 1)
	plt.scatter(round(p_c0, 5), v_c0, label = "p0 = " + str(p_c0), color = "green", zorder = 1)

	plt.xlabel('p', fontweight='bold')
	plt.ylabel('v_' + str(N), fontweight = 'bold')
	plt.legend()
	
	plt.savefig('0p' + 'N' + str(N) + 'MAXITER' + str(MAXITER) + 'EPS' + str(EPS) + '.png')
	plt.close()


if __name__ == "__main__":
	for L in L_:
		N = int(L/2) - (L % 2 == 0) # we consider games up to N-stage
		v_Np = np.empty(N_p) # initializing the array for saving the value as a function of p

		start = time.process_time()			
		for t in range(N_p):
			import time
			s = 0
			if t%10==0: 
				print(t)
			for it in range(MAXITER):
				c = generate_costs(L, p[t])
				value_matrix = calculate_value(L, N, c)
				s = s + value_matrix[N, N]/N
			v_Np[t] = s/MAXITER
			if (t == 0) or (t == N_p/2): print('time is:', time.process_time() - start)			
		determine_critical_probabilities_and_plot(p, N, v_Np)