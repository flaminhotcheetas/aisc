import numpy as np

# Training pattern
X = np.array([-1, 1, 1, 1])

# Create weight matrix using Hebbian learning
W = np.outer(X, X)

print("\nAutoassociative Memory Network")

print("\nTraining Pattern:")
print(X)

print("\nWeight Matrix:")
print(W)

# Test patterns
Test_pat = [
    [-1, 1, 1, 1],      # Original pattern
    [0, 1, 1, 1],       # One missing entry
    [-1, -1, 1, 1],     # One mistake entry
    [0, 0, 1, 1],       # Two missing entries
    [-1, -1, -1, 1]     # Two mistake entries
]

labels = [
    "Original Pattern",
    "One missing entry",
    "One mistake entry",
    "Two missing entries",
    "Two mistake entries"
]

# Required output
required_output = np.array([-1, 1, 1, 1])

# Test each pattern
for label, X_test in zip(labels, Test_pat):

    X_test = np.array(X_test)

    # Calculate net input
    Yin = np.dot(X_test, W)

    # Bipolar activation function
    Y = np.where(Yin > 0, 1,
                 np.where(Yin < 0, -1, 0))

    print("\n" + label)
    print("Input Pattern    :", X_test)
    print("Net Input        :", Yin)
    print("Recalled Pattern :", Y)

    # Check whether the required output is obtained
    if np.array_equal(Y, required_output):
        print("Result           : Correct response")
    else:
        print("Result           : Incorrect response")
