# NN with just one neuron.
### Activation function: Sigmoid
### Loss function: Cross Entropy Loss
### Optimizer: Stochastic Gradient Descent

from numpy import random
import matplotlib.pyplot as plt
from src.Layers import Dense
from src.Optimizer import Gradient_Descent
from src.Activation import Sigmoid

def unit_network(inputs, output, lr, epochs):
    
    print("---------------INPUTS---------------\n")
    print(inputs)
    print("---------------ACTUAL OUTPUT---------------\n")
    print(output)

    wt = [random.uniform(-1.0,1.0,len(output))]
    b = [random.uniform(-1.0,1.0,1)]

    nn = Dense(1, wt, b, Sigmoid)

    optimizer = Gradient_Descent(nn, inputs, output)
    losses = optimizer(lr, epochs, 100)

    plt.plot(range(1,epochs,100),losses)
    plt.show()

    y_pred = [nn(x) for x in inputs]
    print("---------------PREDICTED OUTPUT---------------\n")
    print(y_pred)
