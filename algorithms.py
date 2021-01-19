import numpy as np
import pandas as pd
from utils import *

###############################
#  Algorithmes de résolution  #
###############################

def add_vector(E, vector):
    """
        Ajoute la valeur du vecteur E à l'ensemble
    """
    new_E = []
    for point in E:
        new_E.append([point[0]+vector[0], point[1]+vector[1]])
    return np.array(new_E).astype(type(vector[0]))


def pareto_prog_dyn(E, k):
    """
    """
    n = E.shape[0] # nombre de vecteurs

    # Création de la matrice P des valeurs des points non-dominés
    lines = np.arange(k+1) # lignes = tailles de 0 à k
    columns =[str(E[i]) for i in range(n)] # colonnes = sous-ensembles de tailles 1 à n
    #print(sizes)
    #print(columns)
    P = pd.DataFrame(data=[[None for j in range(n)] for i in range(k+1)],
                     columns=columns)
    P.insert(0, 'Taille', [i for i in range(k+1)])
    P.set_index('Taille', inplace=True)

    # Création de la matrice I des indices des vecteurs des solutions de P
    I = pd.DataFrame(data=[[None for j in range(n)] for i in range(k+1)],
                     columns=columns)
    I.insert(0, 'Taille', [i for i in range(k+1)])
    I.set_index('Taille', inplace=True)

    # initialisation
    P.at[0] = [np.array([[0,0]]) for i in range(n)] # P[0, i] = (0, 0) pour tout  de 1 à n
    I.at[0] = [np.array([[]]) for i in range(n)]

    P.at[1,str(E[0])] = np.array([E[0]]) # P[1, 1] = E[1]
    I.at[1, str(E[0])] = np.array([[0]])

    for i in range(2,k+1):
        P.at[i,str(E[0])] = np.array([]) # P[1, i} = ensemble vide pour tout i > 1
        I.at[i,str(E[0])] = np.array([])

    # Récurrence
    for j in range(1, n): # j = nouveau vecteur
        for i in range(1, k+1): # pour chaque taille
            #print(i,j)
            if (j+1)>=i: # le nombre de vecteurs considérés doit être suffisant pour la taille considérée
                #print('E[i] = ', E[i])
                #print('E[j] =', E[j])
                #print('i-1, j-1 =', P.at[i-1,str(E[j-1])])
                #print('i-1,j-i + j =',add_vector(P.at[i-1,str(E[j-1])], E[j]))
                a = add_vector(P.at[i-1,str(E[j-1])], E[j]) #
                #print('a = ', a)
                b = P.at[i, str(E[j-1])] #
                #print('b =', b)
                c = np.append(a, b).astype(type(E[0]))
                #print('somme', c)
                #print(c.shape[0]/2)
                P.at[i,str(E[j])] = np.array(lexico_pareto_filter(c.reshape(int(c.shape[0]/2),2)))
            else:
                P.at[i,str(E[j])] = np.array([])


    return P, I


#TODO : rajouter matrice des indices correspondant
