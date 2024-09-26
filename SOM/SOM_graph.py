import matplotlib.pyplot as plt
import numpy as np

def grafica_som(inputs,W):
    fig,ax = plt.subplots()
    ax.set(xlim=[-1.5,1.5],ylim=[-1.5,1.5],title='Distribución final de pesos')

    # Patrones de entrada
    ax.scatter(inputs[:,0], inputs[:,1], marker='x', color='grey')  

    # Neuronas
    filas,columnas=W.shape
    x = np.array([[W[i, j][0] for j in range(columnas)] for i in range(filas)])
    y = np.array([[W[i, j][1] for j in range(columnas)] for i in range(filas)])
    ax.scatter(x.ravel(),y.ravel(),marker='o',color='blue')

    # Conexiones entre neuronas:
    for i in range(filas):   
        for j in range(columnas):
            w1 = W[i,j]

            d1 = i + 1          # Me desplazo una fila abajo
            if(d1 < filas):                
                w2 = W[d1,j]
                ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')
            
            d2 = i - 1          # Me desplazo una fila arriba
            if(d2 > -1):                
                w2 = W[d2,j]
                ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')

            d3 = j - 1          # Me desplazo una columna a la izquierda
            if(d3 > -1):
                w2 = W[i,d3]
                ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')

            d4 = j + 1          # Me desplazo una columna a la derecha
            if(d4 < columnas):
                w2 = W[i,d4]
                ax.plot([w1[0],w2[0]],[w1[1],w2[1]],color='black')
    plt.show()