import matplotlib.pyplot as plt
import numpy as np

def generate_vectors(n, m):
    """
    	Génère un ensemble de n vecteurs de dimension 2 où les valeurs sont tirés selon une loi normale
    	d'espérance m et d'écart-type m/4
    """
    return np.random.normal(loc=m, scale=m/4, size=(n,2))

def naive_pareto(vectors):
	n,d = vectors.shape
	mask = np.full(vectors.shape, True, dtype=bool)
	for i in range(n):
		vector1 = vectors[i]
		for vector2 in vectors:
			if vector1[0] > vector2[0] and vector1[1] > vector2[1]:
				mask[i] = np.full((1,d), False, dtype=bool) # point non Pareto-optimal
	pareto = vectors[mask]
	return pareto.reshape((int(pareto.shape[0]/d),d))

def optimized_pareto(vectors):
	n, d = vectors.shape
	mask = np.full(vectors.shape, True, dtype=bool)
	sort_vectors = np.lexsort((vectors[:,1],vectors[:,0]))
	min_1 = vectors[sort_vectors[0],1]
	for i in range(1, n):
		vector = vectors[sort_vectors[i]]
		if vector[1] > min_1:
			mask[sort_vectors[i]] = np.full((1,d), False, dtype=bool) # point non Pareto-optimal
		if vector[1] < min_1:
			min_1 = vector[1]
	pareto = vectors[mask]
	return pareto.reshape((int(pareto.shape[0]/d),d))

def show_pareto(vectors, pareto, title=""):
	plt.figure()
	plt.scatter(vectors[:,0], vectors[:,1], label="data")
	plt.scatter(pareto[:,0], pareto[:,1],color='red', marker='s',label='points Pareto-optimaux')
	plt.title("Front de Pareto "+title)
	plt.xlabel("c1")
	plt.ylabel('c2')
	plt.legend()
	plt.show()