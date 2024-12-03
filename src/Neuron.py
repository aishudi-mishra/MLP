class Neuron:
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def __call__(self, input):
    sum = 0.0
    for i in range(len(self.weights)):
      sum += self.weights[i] * input[i]
    sum += self.bias
    return sum

  def calc_gradient(self, input):
    self.gradient = []
    for i in range(len(input)):
      self.gradient.append(input[i])
    self.gradient.append(1)
    return self.gradient

  def backward(self, input, learning_rate, back_grad):
    self.gradient = self.calc_gradient(input)
    self.weights = [w - learning_rate*dw*bg for w, dw, bg in zip(self.weights, self.gradient[:-1], back_grad[:-1])]
    self.bias = self.bias - learning_rate*self.gradient[-1]*back_grad[-1];