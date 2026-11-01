import numpy as np

np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, legacy='1.13')
R = np.array([
    [0.6, 0.3],
    [0.2, 0.9]
])

S = np.array([
    [1.0, 0.5, 0.3],
    [0.8, 0.4, 0.7]
])

# Max-Min Composition

T_max_min = np.zeros((R.shape[0], S.shape[1]))

for i in range(R.shape[0]):
    for j in range(S.shape[1]):
        values = []

        for k in range(R.shape[1]):
            values.append(min(R[i][k], S[k][j]))

        T_max_min[i][j] = max(values)

print("Max-Min Composition:")
print(T_max_min)


# Max-Product Composition

T_max_product = np.zeros((R.shape[0], S.shape[1]))

for i in range(R.shape[0]):
    for j in range(S.shape[1]):
        values = []

        for k in range(R.shape[1]):
            values.append(R[i][k] * S[k][j])

        T_max_product[i][j] = max(values)

print("\nMax-Product Composition:")
print(T_max_product)