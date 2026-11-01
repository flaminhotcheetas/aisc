import math
import pandas as pd

alpha = 0.25

# Training Data : basically the XOR TT
training_data = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0)
]

# Initial weights: Input -> Hidden Layer
v11 = 0.6
v21 = -0.1
v01 = 0.3

v12 = -0.3
v22 = 0.4
v02 = 0.5

# Initial weights: Hidden -> Output Layer
w1 = 0.4
w2 = 0.1
w0 = -0.2


# Binary Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Store table + mean error for every epoch
all_epoch_tables = []


# RUN BACKPROPAGATION FOR 10 EPOCHS
for epoch in range(1, 11):

    total_error = 0
    epoch_rows = []

    # Process all training patterns
    for x1, x2, t in training_data:

        # 1. HIDDEN LAYER

        zin1 = v01 + x1 * v11 + x2 * v21
        zin2 = v02 + x1 * v12 + x2 * v22

        z1 = sigmoid(zin1)
        z2 = sigmoid(zin2)

        # 2. OUTPUT LAYER

        yin = w0 + z1 * w1 + z2 * w2
        y = sigmoid(yin)

        # 3. ERROR CALCULATION

        error = (t - y) ** 2
        total_error += error

        # 4. OUTPUT ERROR SIGNAL

        delta_out = (t - y) * y * (1 - y)

        # 5. CHANGE IN HIDDEN -> OUTPUT WEIGHTS

        dw1 = alpha * delta_out * z1
        dw2 = alpha * delta_out * z2
        dw0 = alpha * delta_out

        # 6. HIDDEN LAYER ERROR SIGNALS

        # Use old w1 and w2 before updating them
        delta_in1 = delta_out * w1
        delta_in2 = delta_out * w2

        delta1 = delta_in1 * z1 * (1 - z1)
        delta2 = delta_in2 * z2 * (1 - z2)

        # 7. CHANGE IN INPUT -> HIDDEN WEIGHTS

        # Weights connected to Z1
        dv11 = alpha * delta1 * x1
        dv21 = alpha * delta1 * x2
        dv01 = alpha * delta1

        # Weights connected to Z2
        dv12 = alpha * delta2 * x1
        dv22 = alpha * delta2 * x2
        dv02 = alpha * delta2

        # Store values for summary table
        epoch_rows.append({
            "x1": x1,
            "x2": x2,
            "Target": t,
            "Z1": round(z1, 4),
            "Z2": round(z2, 4),
            "Yin": round(yin, 4),
            "Y": round(y, 4),
            "Error": round(error, 4),
            "dw1": round(dw1, 4),
            "dw2": round(dw2, 4),
            "dw0": round(dw0, 4)
        })

        # 8. UPDATE ALL WEIGHTS

        # Input -> Hidden Layer
        v11 = v11 + dv11
        v21 = v21 + dv21
        v01 = v01 + dv01

        v12 = v12 + dv12
        v22 = v22 + dv22
        v02 = v02 + dv02

        # Hidden -> Output Layer
        w1 = w1 + dw1
        w2 = w2 + dw2
        w0 = w0 + dw0

    # Calculate mean error after one complete epoch
    mean_error = total_error / len(training_data)

    # Store table and mean error
    all_epoch_tables.append({
        "table": pd.DataFrame(epoch_rows),
        "mean_error": mean_error
    })


# EPOCH-WISE SUMMARY TABLES

print("\n" + "=" * 90)
print("EPOCH-WISE SUMMARY TABLES")
print("=" * 90)

for i, epoch_data in enumerate(all_epoch_tables, start=1):

    print("\n" + "-" * 90)
    print(f"EPOCH = {i}")
    print("-" * 90)

    # Print table
    print(epoch_data["table"].to_string(index=False))

    # Print mean error
    print(f"\nMean Error = {epoch_data['mean_error']:.4f}")


# FINAL WEIGHTS AFTER 10 EPOCHS

print("\n" + "=" * 60)
print("FINAL WEIGHTS AFTER 10 EPOCHS")
print("=" * 60)

print("\nInput -> Hidden Layer:")

print(f"v11 = {v11:.4f}")
print(f"v21 = {v21:.4f}")
print(f"v01 = {v01:.4f}")

print(f"v12 = {v12:.4f}")
print(f"v22 = {v22:.4f}")
print(f"v02 = {v02:.4f}")

print("\nHidden -> Output Layer:")

print(f"w1 = {w1:.4f}")
print(f"w2 = {w2:.4f}")
print(f"w0 = {w0:.4f}")


# TRAINING COMPLETED

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("Total Epochs = 10")
print("=" * 60)

