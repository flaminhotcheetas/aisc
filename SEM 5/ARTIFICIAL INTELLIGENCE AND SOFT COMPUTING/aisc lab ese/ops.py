import numpy as np

np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, legacy='1.13')

mach = np.array([0.64, 0.645, 0.65, 0.655, 0.66])

A = np.array([0, 0.75, 1, 0.5, 0])
B = np.array([0, 0.25, 0.75, 1, 0.5])

union = np.maximum(A, B)
intersection = np.minimum(A, B)
A_complement = 1 - A
B_complement = 1 - B

print("Mach:", mach)
print("A:", A)
print("B:", B)

print("\na) A ∪ B:", union)
print("\n")
print("b) A ∩ B:", intersection)
print("\n")
print("c) A':", A_complement)
print("\n")
print("d) B':", B_complement)
print("\n")

print("e) A' ∩ B':", np.minimum(A_complement, B_complement))
print("\n")

print("f) A' ∪ B':", np.maximum(A_complement, B_complement))

print("")

