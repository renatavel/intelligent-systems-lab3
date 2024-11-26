import numpy as np
import matplotlib.pyplot as plt

def gaussian_rbf(x, center, radius):
    return np.exp(-((x - center) ** 2) / (2 * radius ** 2))

X = np.linspace(0.1, 1, 20).reshape(-1, 1)
y_target = ((1 + 0.6 * np.sin(2 * np.pi * X / 0.7)) + 0.3 * np.sin(2 * np.pi * X)) / 2

c1, r1 = 0.2, 0.15
c2, r2 = 0.9, 0.15

phi1 = gaussian_rbf(X, c1, r1)
phi2 = gaussian_rbf(X, c2, r2)

Phi = np.hstack([phi1, phi2, np.ones((X.shape[0], 1))])

w = np.random.randn(3, 1)
learning_rate = 0.1
epochs = 2000

for epoch in range(epochs):
    y_pred = Phi.dot(w)
    loss = np.mean((y_pred - y_target) ** 2)
    error = y_pred - y_target
    w_gradient = Phi.T.dot(error) / X.shape[0]
    w -= learning_rate * w_gradient

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

plt.plot(X, y_target, label="Target Output", color="blue")
plt.plot(X, y_pred, label="RBF Network Approximation", color="red", linestyle="--")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.legend()
plt.show()