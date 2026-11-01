import numpy as np

np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, legacy='1.13')
# Membership values of fuzzy sets A and B
A = np.array([0.3, 0.7, 1.0])
B = np.array([0.4, 0.9])

# Cartesian product using minimum operation
R = np.zeros((len(A), len(B)))

for i in range(len(A)):
    for j in range(len(B)):
        R[i][j] = min(A[i], B[j])

print("Fuzzy Set A:", A)
print("Fuzzy Set B:", B)

print("\nCartesian Product R = A x B:")
print(R)