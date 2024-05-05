import numpy as np

vector = np.array([[1, 1, -1, -1],
                   [1, -1, 1, -1],
                   [-1, 1, -1, 1],
                   [-1, -1, 1, 1]])

weights = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        if i == j:
            weights[i][j] = 0
        else:
            weights[i][j] = (np.dot(vector[i], vector[j])) / 4


def activation(x):
    if x >= 0:
        return 1
    else:
        return -1


def hopfield_network(x, weights):
    y = np.copy(x)
    for i in range(4):
        sum = 0
        for j in range(4):
            sum += weights[i][j] * y[j]
        y[i] = activation(sum)
    return y


for i in range(4):
    print("Input Vector:", vector[i])
    output = hopfield_network(vector[i], weights)
    print("Output Vector:", output)