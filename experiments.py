from utils import generate_vectors, update_progress
import matplotlib.pyplot as plt
import numpy as np
import time

######################
#  Expérimentations  #
######################


def naive_vs_sorted(naive_pareto, optimized_pareto, nb_iter=50):
    """
        Compare les temps d'exécutions des algorithmes naive_pareto et optimized_pareto sur nb_iter ensembles de vecteurs générés aléatoirement
    """
    times_naive = []
    times_opti = []
    m = 1000 # espérance de la loi normale pour la génération des vecteurs
    for n in range(200, 10001, 200): # n = nombre de vecteurs
        update_progress(n /(50*200))
        time_naive = time.time()
        for i in range(nb_iter):
            vectors = generate_vectors(n, m)
            pareto = naive_pareto(vectors)
        time_naive = (time.time() - time_naive) / nb_iter

        time_opti = time.time()
        for i in range(nb_iter):
            vectors = generate_vectors(n, m)
            pareto = optimized_pareto(vectors)
        time_opti = (time.time() - time_opti) / nb_iter

        times_naive.append(time_naive)
        times_opti.append(time_opti)

    x = np.arange(200, 10001, 200)
    plt.figure()
    plt.plot(x, times_opti, label='algorithme optimal')
    plt.plot(x, times_naive, label="algorithme naïf")
    plt.title("Comparaison des temps de calcul des points Pareto non dominés")
    plt.xlabel("n")
    plt.ylabel('temps (s)')
    plt.legend()
    plt.show()

    return times_naive, times_opti
