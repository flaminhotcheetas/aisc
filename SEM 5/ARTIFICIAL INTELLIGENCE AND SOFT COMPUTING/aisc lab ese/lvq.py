import numpy as np

# Initial reference vectors
w1 = np.array([0, 0, 1, 1], dtype=float)
w2 = np.array([1, 0, 0, 0], dtype=float)

weights = np.array([w1, w2])

# Classes of reference vectors
weight_classes = np.array([1, 2])

# Training vectors
X = np.array([
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 1, 1, 0]
], dtype=float)

# Actual classes
T = np.array([2, 1, 1])

alpha = 0.1

print("Initial weights:")
print(weights)

for k in range(len(X)):

    x = X[k]
    target = T[k]

    print("\n----------------------------")
    print("Input:", x)
    print("Target class:", target)

    # Calculate Euclidean distance
    distances = []

    for j in range(len(weights)):
        d = np.sum((weights[j] - x) ** 2)
        distances.append(d)

        print(f"D({j+1}) = {d:.4f}")

    # Winner
    winner = np.argmin(distances)

    print("Winning neuron:", winner + 1)
    print("Winner class:", weight_classes[winner])

    # Correct classification
    if weight_classes[winner] == target:

        print("Correct class → move weight TOWARDS input")

        weights[winner] = (
            weights[winner]
            + alpha * (x - weights[winner])
        )

    # Wrong classification
    else:

        print("Wrong class → move weight AWAY from input")

        weights[winner] = (
            weights[winner]
            - alpha * (x - weights[winner])
        )

    print("Updated weights:")
    print(weights)

print("\nFinal weights:")
print(weights)