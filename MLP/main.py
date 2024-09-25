import MLP as MLP
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import functions as f

[x, yD] = f.read_CSV("data/OR_90_trn.csv", 2, 1)
[xP, yP] = f.read_CSV("data/OR_90_tst.csv", 2, 1)

layers = [3,1]
max_epochs = 100
learning_rate = 0.01
error_rate = 0.01
MLP.build(
    x, yD, xP, yP, layers, max_epochs, learning_rate, error_rate
)

plt.figure(3)
plt.title("Data training distribution")
plt.grid()
for i in range(len(x)):

    if yD[i] == 1:
        plt.scatter(x[i,0], x[i,1], color='blue')
    else:
        plt.scatter(x[i,0], x[i,1], color='red')

plt.show()