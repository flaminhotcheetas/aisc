import numpy as np

# Training input vectors
S = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 1]
])

# Target output vectors
T = np.array([
    [0, 1],
    [0, 1],
    [1, 0],
    [1, 0]
])

# Initialize weight matrix
W = np.zeros((4, 2), dtype=int)

# Outer product learning rule
for i in range(len(S)):
    W = W + np.outer(S[i], T[i])

print("\nHeteroassociative Memory Network")

print("\nTraining Input Vectors:")
print(S)

print("\nTarget Output Vectors:")
print(T)

print("\nWeight Matrix:")
print(W)

# Test the network using all training inputs
for i in range(len(S)):

    X = S[i]

    # Calculate net input
    Yin = np.dot(X, W)

    # Binary activation function
    Y = np.where(Yin > 0, 1, 0)

    print("\nTest", i + 1)
    print("Input Pattern   :", X)
    print("Net Input       :", Yin)
    print("Output          :", Y)
    print("Expected Output :", T[i])

    # Check whether output matches expected target
    if np.array_equal(Y, T[i]):
        print("Result          : Correct Response")
    else:
        print("Result          : Incorrect Response")
