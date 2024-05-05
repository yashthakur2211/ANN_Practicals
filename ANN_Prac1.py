import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def linear(x):
    return x


# Generate x values
x = np.linspace(-5, 5, 100)

# Compute y values for each activation function
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_softmax = linear(x)

# Plotting
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, y_tanh, label='Tanh')
plt.title('Hyperbolic Tangent (tanh) Activation Function')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, y_relu, label='ReLU')
plt.title('Rectified Linear Unit (ReLU) Activation Function')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, y_softmax, label='Linear')
plt.title('Linear Activation Function')
plt.legend()

plt.tight_layout()
plt.show()