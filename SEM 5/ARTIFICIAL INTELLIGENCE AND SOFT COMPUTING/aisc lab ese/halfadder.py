def mp_neuron(inputs, weights, threshold):
 activation = sum(x * w for x, w in zip(inputs, weights))
 return 1 if activation >= threshold else 0
# basic gates
def AND_gate(x1, x2):
 return mp_neuron([x1, x2], [1, 1], threshold=2)
def OR_gate(x1, x2):
 return mp_neuron([x1, x2], [1, 1], threshold=1 )
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
# Truth Table
print(f"{'A':<5} {'B':<5} | {'Sum':<5} {'Carry':<5}")
print("-" * 25)
inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
for A, B in inputs:
 S, C = half_adder(A, B)
 print(f"{A:<5} {B:<5} | {S:<5} {C:<5}")