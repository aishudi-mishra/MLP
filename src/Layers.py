from src.Neuron import Neuron

class Dense:
    def __init__(self, num_of_neurons, weights, bias, activation):
        self.neurons = [Neuron(weights[i], bias[i]) for i in range(num_of_neurons)]
        self.activation = activation()

    def __call__(self, input):
        return [self.activation(neuron(input)) for neuron in self.neurons]

    def backward(self, input, learning_rate, back_grad):
        for neuron in self.neurons:
            activation_grad = self.activation.gradient(neuron(input))
            neuron.backward(input, learning_rate, [back_grad*activation_grad])