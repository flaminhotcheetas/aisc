# McCulloch-Pitts Neuron Foundation
def mp_neuron(inputs, weights, threshold):
    activation = sum(x * w for x, w in zip(inputs, weights))
    return 1 if activation >= threshold else 0

# Basic Logic Gates
def AND_gate(x1, x2):
    return mp_neuron([x1, x2], [1, 1], threshold=2)

def OR_gate(x1, x2):
    return mp_neuron([x1, x2], [1, 1], threshold=1)

def AND_NOT_gate(x1, x2):
    return mp_neuron([x1, x2], [1, -1], threshold=1)

def XOR_gate(x1, x2):
    z1 = AND_NOT_gate(x1, x2)
    z2 = AND_NOT_gate(x2, x1)
    return OR_gate(z1, z2)

# Half Adder
def half_adder(A, B):
    Sum = XOR_gate(A, B)
    Carry = AND_gate(A, B)
    return Sum, Carry

# Full Adder
def full_adder(a, b, carry_in):
    sum1, carry1 = half_adder(a, b)
    sum2, carry2 = half_adder(sum1, carry_in)
    carry_out = OR_gate(carry1, carry2)
    return sum2, carry_out

# Truth Table Header
print(f"{'A':<3} {'B':<3} {'C_in':<5} | {'Sum':<5} {'C_out':<5}")
print("-" * 27)

# All 8 possible input combinations for a 3-bit Full Adder
inputs = [
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
]

# Run Truth Table Loop
for A, B, C_in in inputs:
    S, C_out = full_adder(A, B, C_in)
    print(f"{A:<3} {B:<3} {C_in:<5} | {S:<5} {C_out:<5}")