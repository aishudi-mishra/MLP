from src.Loss import Cross_entropy_loss

class Gradient_Descent:
    def __init__(self, nn, inputs, output):
        self.inputs = inputs
        self.output = output
        self.nn = nn
        self.losses = []

    def __call__(self, learning_rate, epochs, step):
        loss = Cross_entropy_loss()
        for epoch in range(epochs):
            total_loss = 0.0
            for x,y in zip(self.inputs, self.output):
                y_pred = self.nn(x)[0]
                loss_grad = loss.gradient(y_pred, y)
                self.nn.backward(x, learning_rate, loss_grad)
                total_loss += loss(y_pred, y)
            if epoch%step==0:
                self.losses.append(total_loss/len(self.output))
        return self.losses