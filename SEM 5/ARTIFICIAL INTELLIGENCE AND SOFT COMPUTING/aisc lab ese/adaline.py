import numpy as np

# Training data for AND Gate (Bipolar inputs and targets)
# x1, x2, bias (fixed to 1), target
data = np.array([
    [1, 1, 1, 1],
    [1, -1, 1, -1],
    [-1, 1, 1, -1],
    [-1, -1, 1, -1]
])

# Initialize weights and bias connection
w = np.zeros(3)  # [w1, w2, b]
alpha = 0.2      # Learning rate
tolerance = 0.5  # Mean Squared Error convergence tolerance
max_epochs = 20

print("Initial weights:", w)

for epoch in range(1, max_epochs + 1):
    mse = 0
    print(f"\n--- Epoch {epoch} ---")
    print("x1\tx2\tt\tw1\tw2\tb\tYin\tError (t-Yin)\tUpdated Weights")
    
    for row in data:
        x = row[:3]
        t = row[3]
        
        # Linear activation (Yin)
        yin = np.dot(x, w)
        
        # Calculate error before thresholding
        error = t - yin
        mse += error ** 2
        
        # Save old weights for logging
        old_w = w.copy()
        
        # LMS / Delta Weight Update Rule
        w += alpha * error * x
        
        print(f"{x[0]:.0f}\t{x[1]:.0f}\t{t:.0f}\t{old_w[0]:.2f}\t{old_w[1]:.2f}\t{old_w[2]:.2f}\t{yin:.2f}\t{error:.2f}\t\t[{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}]")
        
    mse /= len(data)
    print(f"Mean Squared Error: {mse:.4f}")
    
    if mse < tolerance:
        print(f"\nConvergence achieved in {epoch} epochs!")
        break

print("\nFinal ADALINE Parameters:")
print("Weight 1 (w1) =", round(w[0], 4))
print("Weight 2 (w2) =", round(w[1], 4))
print("Bias (b)      =", round(w[2], 4))

# Test the model with final weights
print("\nTesting Final Model:")
for row in data:
    x = row[:3]
    t = row[3]
    yin = np.dot(x, w)
    y = 1 if yin >= 0 else -1
    print(f"Input: {x[:2]} | Net Input: {yin: .2f} | Predicted Output: {y} | Target: {t}")