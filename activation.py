import numpy as np
import matplotlib.pyplot as plt
# updated feature 2 branch


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def tanh(x):
    return np.tanh(x)


# Generate data
x = np.linspace(-10, 10, 100)
sigmoid_y = sigmoid(x)
relu_y = relu(x)
leaky_relu_y = leaky_relu(x)
tanh_y = tanh(x)

# Plot activation functions
plt.figure(figsize=(10, 6))

plt.plot(x, sigmoid_y, label='Sigmoid', color='blue')
plt.plot(x, relu_y, label='ReLU', color='green')
plt.plot(x, leaky_relu_y, label='Leaky ReLU', color='red')
plt.plot(x, tanh_y, label='Tanh', color='orange')

plt.title('Activation Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)

plt.show()
