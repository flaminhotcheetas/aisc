import numpy as np

# Input pattern E
E = np.array([
    1, 1, 1,
    1, -1, -1,
    1, 1, 1,
    1, -1, -1,
    1, 1, 1
])

# Input pattern F
F = np.array([
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    -1, -1, 1,
    -1, -1, 1
])

# Target output patterns
T_E = np.array([-1, 1])
T_F = np.array([1, 1])

# Create weight matrix using outer product rule
W = np.outer(E, T_E) + np.outer(F, T_F)

print("\nBam Network")

print("\nWeight Matrix:")
print(W)

# Test E
Yin_E = np.dot(E, W)

# Bipolar activation function
Y_E = np.where(Yin_E > 0, 1, -1)

print("\nTest E")
print("Input Pattern   :", E)
print("Net Input       :", Yin_E)
print("Output          :", Y_E)
print("Expected Output :", T_E)

if np.array_equal(Y_E, T_E):
    print("Result          : Correct Response")
else:
    print("Result          : Incorrect Response")

# Test F
Yin_F = np.dot(F, W)

# Bipolar activation function
Y_F = np.where(Yin_F > 0, 1, -1)

print("\nTest F")
print("Input Pattern   :", F)
print("Net Input       :", Yin_F)
print("Output          :", Y_F)
print("Expected Output :", T_F)

if np.array_equal(Y_F, T_F):
    print("Result          : Correct Response")
else:
    print("Result          : Incorrect Response")