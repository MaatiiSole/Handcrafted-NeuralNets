import math
import numpy as np
import SOM_train as SOM_t
import SOM_graph as SOM_g
#=========== Estructura del SOM ================================#

filas = 5
columnas = 5

#========= Ordenamiento Global =================================#

mu_ini = 0.9
mitad_mapa = math.ceil(max(filas,columnas)/2)
mu = [mu_ini,mu_ini]
rv = [mitad_mapa,mitad_mapa]
max_epocas = 30

inputs = np.loadtxt('./data/te.csv',delimiter=',')

W = np.empty([filas,columnas], dtype=object)
for i in range(filas):
    for j in range(columnas):
        W[i,j] = np.random.rand(len(inputs[0]))-0.5

W = SOM_t.som_train(inputs,W,max_epocas,rv,mu)
SOM_g.grafica_som(inputs,W)

#========= Etapa de transición =================================#

mu = [mu_ini,0.1]
rv = [mitad_mapa,1]
max_epocas = 300

# No inicializo pesos, uso los resultantes de la etapa anterior.
W = SOM_t.som_train(inputs,W,max_epocas,rv,mu)
SOM_g.grafica_som(inputs,W)

#========= Ajuste fino =================================#

mu = [0.01,0.01]
rv = [0,0]
max_epocas = 50

# No inicializo pesos, uso los resultantes de la etapa anterior.
W = SOM_t.som_train(inputs,W,max_epocas,rv,mu)
SOM_g.grafica_som(inputs,W)     