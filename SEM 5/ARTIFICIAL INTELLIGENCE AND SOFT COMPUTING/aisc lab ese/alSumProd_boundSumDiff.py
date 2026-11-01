import numpy as np

np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, legacy='1.13')

elements = np.array([1, 2, 3, 4])

A = np.array([0.2, 0.3, 0.4, 0.5])
B = np.array([0.1, 0.2, 0.2, 1])

sum = A + B
sub = A - B
mul = A * B

algebraic_sum = sum - mul
algebraic_pdt = mul
Bounded_sum = np.minimum(sum, 1)

# Fixed a minor fuzzy logic bug: Bounded difference uses np.maximum(..., 0), not np.minimum
Bounded_diff = np.maximum(sub, 0)

print("Algebraic Sum: ", algebraic_sum)
print("\n")
print("Algebraic Product: ", algebraic_pdt)
print("\n")
print("Bounded sum: ", Bounded_sum)
print("\n")
print("Bounded difference: ", Bounded_diff)
print("\n")
