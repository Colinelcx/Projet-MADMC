from utils import *
from algorithms import *
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

def compare_procedures(procedure1, procedure2, nb_iter=50):
    """
        Compare les temps d'exécutions des algorithmes naive_pareto et optimized_pareto sur nb_iter ensembles de vecteurs générés aléatoirement
    """
    times_1 = []
    times_2 = []
    n = 50 # taille des instances
    k = 10 # taille des sous-ensembles
    m = 1000 # espérance de la loi normale pour la génération des vecteurs
    for epsilon in np.arange(0.025, 0.5, 0.025): # n = nombre de vecteurs
        update_progress(epsilon /(0.5))

        alpha_min = 0.5 - epsilon
        alpha_max = 0.5 + epsilon

        time_1 = time.time()
        for i in range(nb_iter):
            E = generate_vectors(n, m)
            two_phased_pareto(E, k, alpha_min, alpha_max, show=False)
        time_1 = (time.time() - time_1) / nb_iter

        time_2 = time.time()
        for i in range(nb_iter):
            E = generate_vectors(n, m)
            two_phased_idominance(E, k, alpha_min, alpha_max, show=False)
        time_2 = (time.time() - time_2) / nb_iter

        times_1.append(time_1)
        times_2.append(time_2)

    update_progress(100)
    
    x = np.arange(0.025, 0.5, 0.025)
    plt.figure()
    plt.plot(x, times_1, label='procédure Pareto-dominance')
    plt.plot(x, times_2, label="procédure I-dominance")
    plt.title("Comparaison des temps de calcul des deux procédures")
    plt.xlabel("n")
    plt.ylabel('temps (s)')
    plt.legend()
    plt.show()

    return times_naive, times_opti
