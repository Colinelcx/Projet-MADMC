from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import time


#######################
#  Fonctions de base  #
#######################


def generate_vectors(n, m):
    """
    	Génère un ensemble de n vecteurs de dimension 2 où les valeurs sont tirés selon une loi normale
    	d'espérance m et d'écart-type m/4
    """
    return np.random.normal(loc=m, scale=m/4, size=(n,2))


#######################
#  Filtres de Pareto  #
#######################


def pareto_dominate(vector1, vector2):
    """
        Calcule la dominance de Pareto (au sens fort) de vector1 sur vector2 (vecteurs de dimension 2)
    """
    if vector1[0] < vector2[0] and vector1[1] <= vector2[1]:
        return True
    elif vector1[0] <= vector2[0] and vector1[1] < vector2[1]:
        return True
    else:
        return False

def naive_pareto_filter(vectors):
    """
        Calcule l'ensemble des vecteurs Pareto non-dominés par comparaison par paires systématiques
    """
    t = time.time()
    n,d = vectors.shape
    mask = np.full(vectors.shape, True, dtype=bool)
    for i in range(n):
        vector1 = vectors[i]
        for vector2 in vectors:
            if pareto_dominate(vector2, vector1):
                mask[i] = np.full((1,d), False, dtype=bool) # point Pareto-dominé
    pareto = vectors[mask]
    return pareto.reshape((int(pareto.shape[0]/d),d)), mask

def naive_fast_pareto_filter(vectors):
    """
        Calcule l'ensemble des vecteurs Pareto non-dominés par comparaison par paires
    """
    t = time.time()
    n,d = vectors.shape
    mask = np.full(vectors.shape, True, dtype=bool)
    for i in range(n):
        vector1 = vectors[i]
        for vector2 in vectors:
            if pareto_dominate(vector2, vector1):
                mask[i] = np.full((1,d), False, dtype=bool) # point Pareto-dominé
                break
    pareto = vectors[mask]
    return pareto.reshape((int(pareto.shape[0]/d),d)), mask

def lexico_pareto_filter(vectors):
    """
        Calcule l'ensemble des vecteurs Pareto non-dominés par la méthode de tri
    """
    n, d = vectors.shape
    mask = np.full(vectors.shape, False, dtype=bool)
    sort_vectors = np.lexsort((vectors[:,1],vectors[:,0]))
    mask[sort_vectors[0]] = np.full((1,d), True, dtype=bool) # initialisation
    min_1 = vectors[sort_vectors[0],1]
    for i in range(1, n):
        vector = vectors[sort_vectors[i]]
        if vector[1] < min_1: # point dominant sur le second critère
            min_1 = vector[1]
            mask[sort_vectors[i]] = np.full((1,d), True, dtype=bool) # point non Pareto-dominé
        elif mask[sort_vectors[i-1]][0]: # si le point précédent n'est pas Pareto-dominé
            if vector[0]==vectors[sort_vectors[i-1],0] and vector[1]==vectors[sort_vectors[i-1],1]: # égalité
                mask[sort_vectors[i]] = np.full((1,d), True, dtype=bool) # point non Pareto-dominé
    pareto = vectors[mask]
    return pareto.reshape((int(pareto.shape[0]/d),d)), mask


#######################################################
#  Outils pour le tableau de programmation dynamique  #
#######################################################


def add_P(P, vector):
    """
        Ajoute la valeur du vecteur E à l'ensemble P
    """
    new_P = []
    for point in P:
        new_P.append([point[0]+vector[0], point[1]+vector[1]])
    return np.array(new_P)

def add_I(I, i):
    """
        Ajoute l'élément i à chaque solution de l'ensemble I
    """
    if len(I) == 0:
        return [[i]]
    else :
        new_I = []
        for point in I:
            new_point = point.copy()
            new_point.append(i)
            new_I.append(new_point)
        return new_I

def union_I(left_I, right_I, mask):
    """
        Fusionne les ensembles d'indices gauche et droite en conservant ceux présents dans le masque
    """
    for r in right_I:
        left_I.append(r)
    union = []
    for i in range(mask.shape[0]):
        if mask[i,0]:
            union.append(left_I[i])
    return union

def column_name(E,i):
    """
        Retourne le nom de la ième colonne du tableau de programmation dynamique
        calculant les points non-dominés de l'ensemble E
    """
    return str(i)+":"+str(E[i])


###########################
#  Réduction I-Dominance  #
###########################


def reduce(E, alpha_min, alpha_max):
    """
        Réduit une instance E du problème de détermination des points I-dominés
        en une instance E_reduced du problème de détermination des points non dominés au sens de Pareto
    """
    reduced_E = []
    for y_1, y_2 in E:
        reduced_E.append([y_1 * alpha_min + y_2 * (1 - alpha_min), y_1 * alpha_max + y_2 * (1 - alpha_max)])
    return np.array(reduced_E).astype(float)

def reconstruct(E_reduced, alpha_min, alpha_max,t=float):
    """
        Reconstitue les images des solutions de E à partir des images des solutions de E_reduced
    """
    E = []
    for y1_r, y2_r in E_reduced:
        y1 = round((y1_r * (1 - alpha_max) - y2_r * (1 - alpha_min)) /(alpha_min - alpha_max), 5)
        y2 = round((alpha_min * y2_r - alpha_max * y1_r) / (alpha_min - alpha_max), 5)
        E.append([y1, y2])
    return np.array(E).astype(t)

def reconstruct_P(P_reduced, alpha_min, alpha_max, t=float):
    """
        Reconstruit le tableau de programmation dynamique P des solutions
        à partir du tableau P_reduced des solutions réduites
    """
    return P_reduced.applymap(lambda x: reconstruct(x, alpha_min, alpha_max,t))

def check_transform():
    """
        Vérifie l'exactitude de la réduction/reconstruction sur 1000 vecteurs aléatoires
    """
    alpha_min = 0.1
    alpha_max = 0.8
    n = 1000 # nombre de vecteurs générés
    m = 10
    E = generate_vectors(n,m)
    E_rec = reconstruct(reduce(E, alpha_min, alpha_max), alpha_min, alpha_max)
    assert np.allclose(E, E_rec)


#################################
#   Affichage et visualisation  #
#################################


def show_pareto_front(vectors, pareto, minimax=None, title=""):
    """
        Affiche les vecteurs ainsi que les vecteurs non-dominés
    """
    plt.figure()
    plt.scatter(vectors[:,0], vectors[:,1], label="data")
    plt.scatter(pareto[:,0], pareto[:,1],color='orange', marker='s', label='points Pareto-optimaux')
    if minimax:
        plt.scatter(minimax[0], minimax[1],color='red', marker='s', label='point Minimax')
    plt.title("Front de Pareto "+title)
    plt.xlabel("c1")
    plt.ylabel('c2')
    plt.legend()
    plt.show()

def update_progress(progress):
    """
        Met à jour la barre de progression (pour jupyter notebook)
    """
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
    #clear_output(wait = True)
    text = "Progression: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    if progress == 1:
        print(text, end='\n')
    else:
        print(text, end='\r')
