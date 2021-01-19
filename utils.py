from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import time

def generate_vectors(n, m):
    """
    	Génère un ensemble de n vecteurs de dimension 2 où les valeurs sont tirés selon une loi normale
    	d'espérance m et d'écart-type m/4
    """
    return np.random.normal(loc=m, scale=m/4, size=(n,2))

def naive_pareto_filter(vectors):
	t = time.time()
	n,d = vectors.shape
	mask = np.full(vectors.shape, True, dtype=bool)
	for i in range(n):
		vector1 = vectors[i]
		for vector2 in vectors:
			if vector1[0] > vector2[0] and vector1[1] > vector2[1]:
				mask[i] = np.full((1,d), False, dtype=bool) # point Pareto-dominé
	pareto = vectors[mask]
	return pareto.reshape((int(pareto.shape[0]/d),d))

def naive_fast_pareto_filter(vectors):
	t = time.time()
	n,d = vectors.shape
	mask = np.full(vectors.shape, True, dtype=bool)
	for i in range(n):
		vector1 = vectors[i]
		for vector2 in vectors:
			if vector1[0] > vector2[0] and vector1[1] > vector2[1]:
				mask[i] = np.full((1,d), False, dtype=bool) # point Pareto-dominé
				break
	pareto = vectors[mask]
	return pareto.reshape((int(pareto.shape[0]/d),d))

def lexico_pareto_filter(vectors):
	n, d = vectors.shape
	mask = np.full(vectors.shape, False, dtype=bool)
	sort_vectors = np.lexsort((vectors[:,1],vectors[:,0]))
	mask[sort_vectors[0]] = np.full((1,d), True, dtype=bool) # initialisation
	min_1 = vectors[sort_vectors[0],1]
	for i in range(1, n):
		vector = vectors[sort_vectors[i]]
		if vector[1] < min_1: # point dominant sur le second critère
			min_1 = vector[1]
			mask[sort_vectors[i]] = np.full((1,d), True, dtype=bool) # point no Pareto-dominé
		elif mask[sort_vectors[i]][0]: # le point précédent n'est pas Pareto-dominé
			if vector[0]==vectors[sort_vectors[i-1],0] and vector[1]==vectors[sort_vectors[i-1],1]:
				mask[sort_vectors[i]] = np.full((1,d), True, dtype=bool) # point non Pareto-dominé
	pareto = vectors[mask]
	return pareto.reshape((int(pareto.shape[0]/d),d))

def show_pareto_front(vectors, pareto, title=""):
	plt.figure()
	plt.scatter(vectors[:,0], vectors[:,1], label="data")
	plt.scatter(pareto[:,0], pareto[:,1],color='red', marker='s',label='points Pareto-optimaux')
	plt.title("Front de Pareto "+title)
	plt.xlabel("c1")
	plt.ylabel('c2')
	plt.legend()
	plt.show()

def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progression: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)
