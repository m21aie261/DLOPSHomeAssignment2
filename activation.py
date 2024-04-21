# craeted a feature 2 branch for the merging

import numpy as np
# <<<<<<< feature-2
# import matplotlib.pyplot as plt
# # updated feature 2 branch
# =======
# >>>>>>> main


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

# <<<<<<< feature-2

# def leaky_relu(x, alpha=0.01):
#     return np.where(x > 0, x, alpha * x)


# def tanh(x):
#     return np.tanh(x)


# # Generate data
# x = np.linspace(-8, 8, 100)
# sigmoid_y = sigmoid(x)
# relu_y = relu(x)
# leaky_relu_y = leaky_relu(x)
# tanh_y = tanh(x)

# # Plot activation functions
# plt.figure(figsize=(10, 6))

# plt.plot(x, sigmoid_y, label='Sigmoid', color='blue')
# plt.plot(x, relu_y, label='ReLU', color='green')
# plt.plot(x, leaky_relu_y, label='Leaky ReLU', color='red')
# plt.plot(x, tanh_y, label='Tanh', color='orange')

# plt.title('Activation Functions')
# plt.xlabel('Input')
# plt.ylabel('Output')
# plt.legend()
# plt.grid(True)

# plt.show()
# =======
print("Sigmoid values for random data:")
for value in random_values:
    print(f"Sigmoid({value}) = {sigmoid(value)}")
# >>>>>>> main
