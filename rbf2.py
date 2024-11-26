import numpy as np
import matplotlib.pyplot as plt

def gaussian_rbf(x, center, radius):
    return np.exp(-((x - center) ** 2) / (2 * radius ** 2))

X = np.linspace(0.1, 1, 20).reshape(-1, 1)
y_target = ((1 + 0.6 * np.sin(2 * np.pi * X / 0.7)) + 0.3 * np.sin(2 * np.pi * X)) / 2


n_rbfs = 5  
centers = np.linspace(0.1, 1, n_rbfs)  # Evenly distribute initial centers
radii = np.full(n_rbfs, 0.2) 


def compute_design_matrix(X, centers, radii):
    Phi = np.hstack([gaussian_rbf(X, c, r).reshape(-1, 1) for c, r in zip(centers, radii)])
    Phi = np.hstack([Phi, np.ones((X.shape[0], 1))])  
    return Phi

Phi = compute_design_matrix(X, centers, radii)

w = np.random.randn(n_rbfs + 1, 1)  
learning_rate = 0.05
epochs = 2000

for epoch in range(epochs):
    y_pred = Phi.dot(w)

    loss = np.mean((y_pred - y_target) ** 2)

    error = y_pred - y_target
    w_gradient = Phi.T.dot(error) / X.shape[0]
    w -= learning_rate * w_gradient

    
    for i in range(n_rbfs):
        phi_i = gaussian_rbf(X, centers[i], radii[i]).reshape(-1, 1)
        center_gradient = np.sum(error * phi_i * (X - centers[i]) / (radii[i] ** 2)) / X.shape[0]
        radius_gradient = np.sum(error * phi_i * ((X - centers[i]) ** 2) / (radii[i] ** 3)) / X.shape[0]

        centers[i] -= learning_rate * center_gradient
        radii[i] -= learning_rate * radius_gradient
        radii[i] = max(radii[i], 0.01)  

    Phi = compute_design_matrix(X, centers, radii)

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
plt.plot(X, y_target, label="Target Output", color="blue")
plt.plot(X, y_pred, label="RBF Network Approximation", color="red", linestyle="--")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.legend()
plt.title("RBF Network Approximation with Adaptive Centers and Radii")
plt.show()