import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for linear regression: y = 2.5*x + 1.0 + noise
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = 2.5 * x + 1.0 + np.random.randn(x.shape[0]) * 2

# Gradient descent settings
w0, w1 = 0.0, 0.0           # initial intercept and slope
learning_rate = 0.001
epochs = 500

n = x.shape[0]
w0_history, w1_history, loss_history = [], [], []

# Gradient descent loop
for epoch in range(epochs):
    y_pred = w1 * x + w0
    error = y - y_pred
    # Compute MSE loss with 1/(2n) factor
    loss = (1/(2*n)) * np.sum(error**2)
    # Compute gradients
    grad_w0 = -(1/n) * np.sum(error)
    grad_w1 = -(1/n) * np.sum(error * x)
    # Update parameters
    w0 -= learning_rate * grad_w0
    w1 -= learning_rate * grad_w1
    
    # Store history
    w0_history.append(w0)
    w1_history.append(w1)
    loss_history.append(loss)

# Print final parameters
print(f"Converged to w0 = {w0:.4f}, w1 = {w1:.4f}")

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1) Data and fitted line
axes[0].scatter(x, y, label='Data')
axes[0].plot(x, w1 * x + w0, color='red', label='Fitted line')
axes[0].set_title('Data & Fitted Regression Line')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].legend()

# 2) Loss over iterations
axes[1].plot(loss_history)
axes[1].set_title('Loss over Iterations')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MSE Loss')

# 3) Parameter trajectories
axes[2].plot(w0_history, label='w0 (intercept)')
axes[2].plot(w1_history, label='w1 (slope)')
axes[2].set_title('Parameter Convergence')
axes[2].set_xlabel('Epoch')
axes[2].legend()

plt.tight_layout()
plt.show()
