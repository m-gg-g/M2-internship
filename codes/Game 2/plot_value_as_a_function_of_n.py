import numpy as np 
import matplotlib.pyplot as plt
from main_functions import generate_costs, calculate_value

MAXITER = 30
p = 0.95
# L_ = np.linspace(11, 201, num = 100, dtype = int)
L_ = [10, 20, 40, 60, 80, 100, 200, 400, 600]

def plot_value_v_n_as_a_function_of_n(N, v, p):
	v_n = v[N]
	x = np.arange(1, N + 1, 1)

	plt.plot(x, [v[i] for i in x])
	
	plt.title("p = " + str(p))
	plt.xlabel('n')
	plt.ylabel('v_n')

	# plt.show()
	plt.savefig('n' + "N" + str(N) + 'p' + str(int(p*100)) + '.png') 
	# plt.close()

if __name__ == "__main__":	
	for L in L_:  
		N = int(L/2) - (L % 2 == 0) # we play until the N-stage game
		v = np.zeros(N + 1) 

		for t in range(MAXITER):
			c = generate_costs(L, p) 
			value_matrix = calculate_value(L, N, c)
			v[1:] = v[1:] + np.divide(np.diag(value_matrix)[1:(N+1)], 2*np.arange(1,N+1))
		v = v/MAXITER
		print(L)

		plot_value_v_n_as_a_function_of_n(N, v, p)
