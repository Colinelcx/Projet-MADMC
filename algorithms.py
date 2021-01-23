import numpy as np
import pandas as pd
from utils import *


#############################
#  Programmation dynamique  #
#############################


def pareto_prog_dyn(E, k, show=True):
    """
        Calcule les sous ensembles Pareto-optimaux de taille k d'un ensemble E de taille n par programmation dynamique
        Retourne :
        - P : tableau des images des solutions non-dominées
        - I : tableau d'indices des vecteurs des solutions non-dominés (= espace des critères)
    """
    n = E.shape[0] # nombre de vecteurs

    lines = np.arange(k+1) # lignes = tailles de 0 à k
    columns =[column_name(E,i) for i in range(n)] # colonnes = sous-ensembles de tailles 1 à n

    # Création de la matrice P des valeurs des points non-dominés
    P = pd.DataFrame(data=[[None for j in range(n)] for i in range(k+1)],
                     columns=columns)
    P.insert(0, 'Taille', [i for i in range(k+1)])
    P.set_index('Taille', inplace=True)

    # Création de la matrice I des antécédents des solutions de P
    I = pd.DataFrame(data=[[None for j in range(n)] for i in range(k+1)],
                     columns=columns)
    I.insert(0, 'Taille', [i for i in range(k+1)])
    I.set_index('Taille', inplace=True)

    # initialisation
    P.at[0] = [np.array([[0,0]]) for i in range(n)] # P[0, i] = (0, 0) pour tout  de 1 à n
    I.at[0] = [[] for i in range(n)] # I[0, i] = [] pour tout  de 1 à n
    P.at[1,column_name(E,0)] = np.array([E[0]]) # P[1, 1] = E[1]
    I.at[1, column_name(E,0)] = [[0]] # I[1, 1] = [0]

    for i in range(2,k+1):
        P.at[i,column_name(E,0)] = np.array([]) # P[1, i] = ensemble vide pour tout i > 1
        I.at[i,column_name(E,0)] = [] # I[1, i] = ensemble vide pour tout i > 1

    # Récurrence
    for j in range(1, n): # j = nouveau vecteur
        if show:
            update_progress(((j-1)*k)/(k*n))
        for i in range(1, k+1): # pour chaque taille
            if (j+1)>=i: # le nombre de vecteurs considérés doit être suffisant pour la taille considérée
                left_P = add_P(P.at[i-1,column_name(E,j-1)], E[j]) # terme de gauche : E[i-1, j-1] auquels on ajoute E[j]
                left_I = add_I(I.at[i-1,column_name(E,j-1)], j)
                right_P = P.at[i,column_name(E,j-1)] # terme de droite : E[E, j-1]
                right_I = I.at[i,column_name(E,j-1)]
                union_P = np.append(left_P, right_P) # somme des termes gauche et droite
                union_P = union_P.reshape(int(union_P.shape[0]/2),2).astype(type(E[0]))
                pareto_points, mask = lexico_pareto_filter(union_P) # filtre de pareto sur l'ensemble formé
                P.at[i,column_name(E,j)] = np.array(pareto_points).astype(E[0,0])
                I.at[i,column_name(E,j)] = union_I(left_I, right_I, mask)
            else:
                P.at[i,column_name(E,j)] = np.array([])
                I.at[i,column_name(E,j)] = []
    if show:
        update_progress(100)
    return P, I


###############################
#  Algorithmes de résolution  #
###############################


def minimax(E, alpha_min, alpha_max, show):
    """
        Renvoie une solution minimax dans l'ensemble E pour l'intervalle [alpha_min, alpha_max]
    """
    minimax = None, 2*E.max()*alpha_max, -1
    n = E.shape[0]
    for i in range(n):
        if show:
            update_progress(i/n)
        y_1, y_2 = E[i]
        if y_1 >= y_2:
            temp = y_1 * alpha_max + y_2 * (1 - alpha_max)
        else:
            temp = y_1 * alpha_min + y_2 * (1 - alpha_min)
        if temp < minimax[1]:
            minimax = (y_1, y_2), temp, i
    if show:
        update_progress(100)
    return minimax

def two_phased_pareto(E, k, alpha_min, alpha_max, show=True):
    """
        Méthode en deux phases :
            - 1 : Détermination des sous ensembles Pareto non-dominés de taille k par programmation dynamique
            - 2 : Détermination d'un point minimax parmis ces sous-ensembles
        Retourne :
            - best_sol : image de la solution minimax
            - sets_i[indice] : antécédent de la solution minimax
            - sets : images des sous-ensembles de taille k non-dominés au sens de Pareto
            - sets_i : antécédents des sous-ensembles de taille k non-dominés au sens de Pareto
            - P : tableau des images des solutions non-dominées dans l'espace des objectifs
            - I : tableaux des solutions non-dominés dans l'espace des objets
    """
    n = E.shape[0]
    if show:
        print("---- Première Phase ----")
    P, I = pareto_prog_dyn(E, k, show)
    pareto_front = P[column_name(E,n-1)][k] # ensembles Pareto-optimaux de taille k dans E
    pareto_front_i = I[column_name(E,n-1)][k]
    if show:
        print("---- Seconde Phase ----")
    minimax_sol, value, indice = minimax(pareto_front, alpha_min, alpha_max, show) #solution minimax
    return minimax_sol, pareto_front_i[indice], pareto_front, pareto_front_i, P, I

def two_phased_idominance(E, k, alpha_min, alpha_max, show=True):
    """
        Méthode en deux phases :
            - 1 : Détermination des sous ensembles non I-dominés de taille k par programmation dynamique
            - 2 : Détermination d'un point minimax parmis ces sous-ensembles
        Retourne :
            - best_sol : image de la solution minimax
            - sets_i[indice] : antécédent de la solution minimax
            - sets : images des sous-ensembles de taille k non I-dominés
            - sets_i : antécédents des sous-ensembles de taille k non I-dominés
            - P : tableau des images des solutions non-dominées dans l'espace des objectifs
            - I : tableaux des solutions non I-dominés dans l'espace des objets
    """
    n = E.shape[0]
    if show:
        print("---- Première Phase ----")
    E_reduced = reduce(E, alpha_min, alpha_max) # réduction du problème I-dominance
    P_reduced, I = pareto_prog_dyn(E_reduced, k, show)
    pareto_front_reduced = P_reduced[column_name(E_reduced,n-1)][k] # ensembles Pareto-optimaux de taille k dans E
    idominate_front_i = I[column_name(E_reduced,n-1)][k]
    idominate_front = reconstruct(pareto_front_reduced, alpha_min, alpha_max)
    if show:
        print("---- Seconde Phase ----")
    minimax_sol, value, indice = minimax(idominate_front, alpha_min, alpha_max, show) #solution minimax
    return minimax_sol, idominate_front_i[indice], idominate_front, idominate_front_i, P_reduced, I
