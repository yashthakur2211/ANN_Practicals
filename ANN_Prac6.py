import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weight1 = np.random.randn(self.input_size, self.hidden_size)
        self.weight2 = np.random.randn(self.hidden_size, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        self.z = np.dot(x, self.weight1)
        self.z2 = self.sigmoid(self.z)

        self.z3 = np.dot(self.z2, self.weight2)
        output = self.sigmoid(self.z3)
        return output

    def backward(self, x, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.derivative(output)

        self.z2_error = self.output_delta.dot(self.weight2.T)
        self.z2_delta = self.z2_error * self.derivative(self.z2)

        self.weight1 += X.T.dot(self.z2_delta)
        self.weight2 += self.z2.T.dot(self.output_delta)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)

            self.backward(X, y, output)

nn = NeuralNetwork(input_size=2,hidden_size =3 ,output_size =1)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

nn.train(X,y,epochs=10000)

new_data = np.array([[0,0.3],[0,0.8],[1,0.2],[1,0.6]])
predictions = nn.forward(new_data)
print(predictions)