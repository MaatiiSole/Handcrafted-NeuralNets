import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions as f


def build(
    inputs, real_output , test_inputs, real_test_output,
    layers, max_epochs, learning_rate, error_rate
):
    # ================================= init variables ================================
    inputs = np.insert(inputs, 0, -1, axis=1)
    number_layers = len(layers)
    SIGMOIDE = 1
    W = np.empty(number_layers, dtype=object)
    epochs = 0
    number_errors = 0
    actual_error_rate = 0
    errors_for_epoch = np.zeros(max_epochs)
    mse_for_epoch = np.zeros(max_epochs)
    flag = True 
    # ============================== Weight inicialitation ================================
    number_inputs = len(inputs[0])
    for c in range(number_layers):
        if c == 0:
            W[c] = np.random.uniform(
                -0.5, 0.5, [layers[c], number_inputs]
            )  # first matrix case
        else:
            W[c] = np.random.uniform(
                -0.5, 0.5, [layers[c], layers[c - 1] + 1]
            )  # +1 for bias

    while epochs < max_epochs and flag == True:  # loop for epoch
        for n in range(len(inputs)):  # loop for sample
            Y = []
            # ========================== Forward propagation ==========================
            x = inputs[n].reshape(-1, 1)
            for i in range(number_layers):  # loop for layer
                lineal_output = W[i] @ x
                y = f.sigmoid(lineal_output, SIGMOIDE)
                y = y.flatten().reshape(-1, 1)
                Y.append(y)

                x = Y[-1]
                x = np.insert(x, 0, -1).reshape(
                    -1, 1
                )
            
            e_j = real_output[n].reshape(-1, 1) - Y[-1]
            deltas = []
            # ========================== Backward propagation ==========================
            for i in range(number_layers - 1, -1, -1):
                d_sigmoid = (1 / 2) * (1 + Y[i]) * (1 - Y[i])
                if i == number_layers - 1:
                    deltas.insert(0, d_sigmoid * e_j)  # last delta (simple case)
                else:
                    weight = W[i + 1]  # Obtenemos los weight de la capa siguiente
                    weight = np.transpose(weight[:, 1:])
                    a = weight @ deltas[0]
                    deltas.insert(0, d_sigmoid * a)

            # ==========================  Weight actualization ==========================

            for i in range(number_layers):
                delta_w = np.zeros_like(W[i])
                if i == 0:
                    x_0 = inputs[n].reshape(-1, 1)
                    delta_w = learning_rate * (deltas[i] @ x_0.T)
                else:
                    y_bias = (
                        np.insert(Y[i - 1], 0, -1).flatten().reshape(-1, 1).T
                    )
                    delta_w = learning_rate * (deltas[i] @ y_bias)
                W[i] = W[i] + delta_w
        # ========================== Test for stop training ==========================
        number_errors = 0
        mse_error = 0

        for k in range(len(inputs)):
            Y = []
            x = inputs[k].reshape(-1, 1) 
            for i in range(number_layers):
                lineal_output = W[i] @ x
                y = f.sigmoid(lineal_output, SIGMOIDE)
                y = y.flatten().reshape(-1, 1)
                Y.append(y)
                
                x = Y[-1]
                x = np.insert(x, 0, -1).reshape(
                    -1, 1
                )  
                
            erorr_aux = f.error_test(real_output[k], Y[-1])
            number_errors += erorr_aux

            mse_error += np.sum((real_output[k].reshape(-1, 1) - Y[-1]) ** 2) / len(real_output[k])
            
        errors_for_epoch[epochs] = number_errors
        mse_for_epoch[epochs] = mse_error
        actual_error_rate = number_errors / len(inputs)
        
        if epochs % 100 == 0 and epochs != 0:
            print("epochs:", epochs)
            print("eror: ", actual_error_rate, "%")
            
        if actual_error_rate < error_rate:
            flag = False

        epochs += 1
    

    print("Number of epochs: ", epochs)
    cantEpocasTr = epochs
    print("Number of error traning for epochs: ", errors_for_epoch)
    cantErroresTr = number_errors
    print("Last error training: ", actual_error_rate)
    errorPorcentualTr = actual_error_rate
    epocas = np.arange(1, epochs + 1)
    fig1 = plt.figure(1)
    plt.plot(epocas, mse_for_epoch[0:epochs])
    plt.title("MSE for epoch")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.grid()
    fig2 = plt.figure(2)
    plt.plot(epocas, errors_for_epoch[0:epochs])
    plt.title("Number of errors for epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Number of errors")
    plt.grid()
    
    plt.show()

    # ========================== Test ==========================
    test_inputs = np.insert(test_inputs, 0, -1, axis=1)
    number_error_test = 0

    for k in range(len(test_inputs)):
        Y = []
        x = test_inputs[k].reshape(-1, 1) 
        for i in range(number_layers):  
            lineal_output = W[i] @ x 
            y = f.sigmoid(lineal_output, SIGMOIDE)
            y = y.flatten().reshape(-1, 1)
            Y.append(y) 

            x = Y[-1]
            x = np.insert(x, 0, -1).reshape(
                -1, 1
            ) 
        erorr_aux = f.error_test(real_test_output[k], Y[-1])
        number_error_test += erorr_aux
    error_rate_test = number_error_test / len(test_inputs)
    print("------------------TESTEO--------------------")
    print("Number of errors in test: ", number_error_test)
    print("Error rate: ", error_rate_test)
    return [
        cantEpocasTr,
        cantErroresTr,
        errorPorcentualTr,
        number_error_test,
        error_rate_test,
    ]
