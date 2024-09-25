import numpy as np


def read_CSV(file_name, number_of_imput, number_of_output):
    """
    read a CSV file
    """
    data = np.loadtxt(file_name, delimiter=",")
    x = np.array(data[:, 0:number_of_imput])
    y = np.array(data[:, number_of_imput : number_of_imput+ number_of_output])
    
    return [x, y]

def sigmoid(v, b):
    a = 0
    y = 2 / (1 + np.exp(-b * v)) - 1
    return y

def error_test(real_y, Y):
    number_errors = 0
    if len(Y) == 1:
        error = real_y - signo(Y)
        if np.any(error != 0):
            number_errors = 1
    else:  # Se elige al mayor como la salida del clasificador
        indiceMayor = np.argmax(
            Y
        )  # miro los dos indices máximos y espero que coincidan
        indiceMayorEsperada = np.argmax(real_y)
        if indiceMayor != indiceMayorEsperada:
            number_errors = 1
        # if not((yD == signo(Y)).all()):
        #     number_errors = 1
    return number_errors

def signo(x):
    aux = 1000
    if isinstance(x,np.ndarray):
        aux = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] >= 0:
                aux[i] = 1  
            else:
                aux[i] = -1
    else:
        if x >= 0:
            aux = 1  
        else:
            aux = -1
    return aux