import numpy as np

class Relu:
    def __call__(self, input):
      return max(0, input)

    def gradient(self, input):
      return input if input > 0 else 0

class Sigmoid:
    def __call__(self, input):
        return (1/(1+np.exp(-input)))[0]

    def gradient(self, input):
        sigma = self.__call__(input)
        return sigma * (1 - sigma)

class Tanh:
    def __call__(self, input):
        e2x = np.exp(2*input)
        return (e2x - 1)/(e2x + 1)

    def gradient(self, input):
        tanh = self.__call__(input)
        return 1 - tanh**2