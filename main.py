import numpy as np

# XOR Data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Seed for reproducibility
np.random.seed(42)

# Initialize sizes
input_size = 2
hidden_size = 2
output_size = 1

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Forward pass
def forward(X):
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# Loss function
def compute_loss(y_true, y_pred):
    epsilon = 1e-10  # Avoid log(0)
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

# Backpropagation and weight update
def backward(X, y, z1, a1, z2, a2, learning_rate=0.1):
    global W1, b1, W2, b2
    m = y.shape[0]
    
    dz2 = a2 - y
    dW2 = a1.T @ dz2 / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    dz1 = (dz2 @ W2.T) * sigmoid_derivative(z1)
    dW1 = X.T @ dz1 / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# Training loop
def train(epochs=10000, learning_rate=0.1):
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(X)
        loss = compute_loss(y, a2)
        backward(X, y, z1, a1, z2, a2, learning_rate)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Run training and show predictions
if __name__ == "__main__":
    train()
    _, _, _, predictions = forward(X)
    print("\nFinal Predictions (rounded):\n", predictions.round())
