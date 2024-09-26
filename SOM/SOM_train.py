import numpy as np
import copy
import matplotlib.pyplot as plt

def som_train(inputs,W,max_epocas,v_rv,v_mu):
    W_last = copy.deepcopy(W)
    dist = np.zeros_like(W)
    filas,columnas=W.shape

    #steps para la etapa de transicion
    step_mu = (v_mu[0] - v_mu[1])/max_epocas
    step_rv = (v_rv[0] - v_rv[1])/max_epocas

    for epoca in range(max_epocas):
        for input in inputs:

            #Velocidad de aprendizaje y radio de vecindad
            mu = v_mu[0] - epoca*step_mu
            rv = round(v_rv[0] - epoca*step_rv)     

            # Calculo las distancias de cada neurona a la entrada y obtengo la ganadora:
            for fila in range(filas):
                for columna in range(columnas):
                    dist[fila,columna] = np.linalg.norm(input - W[fila,columna])

            indice = np.unravel_index(np.argmin(dist),W.shape)
            
            # Bucle para recorrer la matriz de neuronas por las vecindades
            for k in range(0,rv+1):                 # Empieza en 0 para hacer la fila de la neurona ganadora      
                d = indice[0] + k                   # Me desplazo una fila abajo
                if (d < filas):
                    W[d,indice[1]] += mu*(input - W[d,indice[1]])
                    for j in range(1,rv+1-k):       # En esa fila, voy a la izquierda
                        dL = indice[1] - j
                        if(dL > -1): W[d,dL] += mu*(input - W[d,dL])
                        else: break
                    for j in range(1,rv+1-k):       # En esa fila, voy a la derecha
                        dR = indice[1] + j
                        if(dR < columnas): W[d,dR] += mu*(input - W[d,dR])
                        else: break
                else: break
            
            for k in range(1,rv+1):               # Empieza en 1 porque ya hice la fila de la neurona ganadora antes
                d = indice[0] - k                 # Me desplazo una fila arriba
                if (d > -1):
                    W[d,indice[1]] += mu*(input - W[d,indice[1]])
                    for j in range(1,rv+1-k):       # En esa fila, voy a la izquierda
                        dL = indice[1] - j
                        if(dL > -1): 
                            W[d,dL] += mu*(input - W[d,dL])
                        else: break
                    for j in range(1,rv+1-k):       # En esa fila, voy a la derecha
                        dR = indice[1] + j
                        if(dR < columnas): W[d,dR] += mu*(input - W[d,dR])
                        else: break
                else: break

        W_last = copy.deepcopy(W)
        epoca += 1
    
    print('El entrenamiento finalizó en la época',epoca)

    return W_last