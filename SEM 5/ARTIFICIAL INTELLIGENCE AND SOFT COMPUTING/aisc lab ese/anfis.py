# %pip install numpy pandas scikit-learn scipy matplotlib


# EXPERIMENT 10
# ANFIS Model for Traffic Flow Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import least_squares

# --------------------------------------------------
# 1. DEFINE THE PROBLEM AND GENERATE DATA
# --------------------------------------------------

np.random.seed(42)

n_samples = 600

# Traffic density: vehicles per kilometre
density = np.random.uniform(5, 100, n_samples)

# Average speed decreases as traffic density increases
speed = 85 - 0.7 * density + np.random.normal(0, 5, n_samples)
speed = np.clip(speed, 5, 90)

# Traffic flow = density * speed
# Add small noise to simulate variations in traffic
flow = density * speed + np.random.normal(0, 80, n_samples)
flow = np.maximum(flow, 0)

data = pd.DataFrame({
    "Density": density,
    "Speed": speed,
    "Traffic_Flow": flow
})

print("First five rows of the dataset:")
print(data.head())

# Input features and target
X = data[["Density", "Speed"]].values
y = data["Traffic_Flow"].values

# Scale inputs and output for stable training
X = X / 100.0
y = y / 1000.0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 2. DEFINE GAUSSIAN MEMBERSHIP FUNCTIONS
# --------------------------------------------------

# Three membership functions per input:
# Low, Medium, High

n_mfs = 3

initial_centers = np.array([
    0.2, 0.5, 0.8,   # Density membership function centers
    0.2, 0.5, 0.8    # Speed membership function centers
])

initial_sigmas = np.full(6, 0.22)

initial_premise = np.concatenate([
    initial_centers, initial_sigmas
])


def gaussian_mf(x, center, sigma):
    """Calculate Gaussian membership degree."""
    return np.exp(
        -0.5 * ((x - center) / sigma) ** 2
    )


# --------------------------------------------------
# 3. GENERATE FUZZY RULES AND FIRING STRENGTHS
# --------------------------------------------------

def calculate_firing_strengths(X, premise):
    centers = premise[:6].reshape(2, 3)
    sigmas = premise[6:].reshape(2, 3)

    mu_density = np.column_stack([
        gaussian_mf(X[:, 0], centers[0, j], sigmas[0, j])
        for j in range(3)
    ])

    mu_speed = np.column_stack([
        gaussian_mf(X[:, 1], centers[1, j], sigmas[1, j])
        for j in range(3)
    ])

    # 3 membership functions for each of 2 inputs
    # Total rules = 3 * 3 = 9
    firing = np.einsum("bi,bj->bij",
                       mu_density, mu_speed).reshape(len(X), -1)

    # Normalize firing strengths
    normalized_firing = firing / (
        firing.sum(axis=1, keepdims=True) + 1e-12
    )

    return normalized_firing


# --------------------------------------------------
# 4. SUGENO INFERENCE AND CONSEQUENT PARAMETERS
# --------------------------------------------------

def build_design_matrix(X, premise):
    firing = calculate_firing_strengths(X, premise)

    # First-order Sugeno rule output:
    # f = a * density + b * speed + c
    #
    # Each rule has 3 consequent parameters.
    local_outputs = np.column_stack([
        X[:, 0],
        X[:, 1],
        np.ones(len(X))
    ])

    # Weight each rule's output by its normalized firing strength
    design = (
        firing[:, :, None] * local_outputs[:, None, :]
    ).reshape(len(X), -1)

    return design


def predict_anfis(X, premise, consequent):
    design = build_design_matrix(X, premise)
    return design @ consequent


# --------------------------------------------------
# 5. TRAIN THE ANFIS MODEL
# --------------------------------------------------

# Hybrid learning:
# A. Optimize membership function parameters
# B. Estimate consequent parameters using least squares

def objective(premise):
    design = build_design_matrix(X_train, premise)

    # Least-squares estimation of Sugeno consequents
    consequent = np.linalg.lstsq(
        design, y_train, rcond=None
    )[0]

    predictions = design @ consequent

    return predictions - y_train


# Keep centers and widths within sensible ranges
lower_bounds = np.concatenate([
    np.zeros(6),
    np.full(6, 0.05)
])

upper_bounds = np.concatenate([
    np.ones(6),
    np.full(6, 0.8)
])

print("\nTraining ANFIS model...")

result = least_squares(
    objective,
    initial_premise,
    bounds=(lower_bounds, upper_bounds),
    max_nfev=100
)

trained_premise = result.x

# Estimate final consequent parameters
train_design = build_design_matrix(X_train, trained_premise)

trained_consequent = np.linalg.lstsq(
    train_design, y_train, rcond=None
)[0]

print("Training completed.")
print("Optimization successful:", result.success)
print("Number of fuzzy rules:", n_mfs * n_mfs)


# --------------------------------------------------
# 6. TEST THE MODEL
# --------------------------------------------------

y_pred = predict_anfis(
    X_test, trained_premise, trained_consequent
)

# Convert predictions back to vehicles per hour
actual_flow = y_test * 1000
predicted_flow = y_pred * 1000

mae = mean_absolute_error(actual_flow, predicted_flow)

rmse = np.sqrt(
    mean_squared_error(actual_flow, predicted_flow)
)

r2 = r2_score(actual_flow, predicted_flow)

print("\nMODEL PERFORMANCE")
print("---------------------------")
print(f"Mean Absolute Error : {mae:.2f} vehicles/hour")
print(f"Root Mean Square Error: {rmse:.2f} vehicles/hour")
print(f"R2 Score            : {r2:.4f}")


# --------------------------------------------------
# 7. DISPLAY SAMPLE PREDICTIONS
# --------------------------------------------------

results_df = pd.DataFrame({
    "Actual Flow": actual_flow,
    "Predicted Flow": predicted_flow
})

print("\nSample Predictions:")
print(results_df.head(10).round(2))


# --------------------------------------------------
# 8. VISUALIZE ACTUAL VS PREDICTED TRAFFIC FLOW
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    actual_flow, predicted_flow,
    alpha=0.7, label="Predictions"
)

min_value = min(actual_flow.min(), predicted_flow.min())
max_value = max(actual_flow.max(), predicted_flow.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    "r--", label="Perfect prediction"
)

plt.xlabel("Actual Traffic Flow (vehicles/hour)")
plt.ylabel("Predicted Traffic Flow (vehicles/hour)")
plt.title("ANFIS Traffic Flow Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 9. PLOT TRAINING DATA AND PREDICTIONS
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    actual_flow[:50],
    marker="o",
    label="Actual Traffic Flow"
)

plt.plot(
    predicted_flow[:50],
    marker="x",
    label="Predicted Traffic Flow"
)

plt.xlabel("Test Sample")
plt.ylabel("Traffic Flow (vehicles/hour)")
plt.title("Actual vs Predicted Traffic Flow")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 10. PREDICT FLOW FOR A NEW TRAFFIC CONDITION
# --------------------------------------------------

new_density = 40       # vehicles/km
new_speed = 50         # km/hour

new_input = np.array([
    [new_density / 100, new_speed / 100]
])

new_prediction = predict_anfis(
    new_input, trained_premise, trained_consequent
)[0] * 1000

print("\nNEW TRAFFIC CONDITION")
print("---------------------------")
print("Traffic density:", new_density, "vehicles/km")
print("Average speed:", new_speed, "km/hour")
print(
    f"Predicted traffic flow: {new_prediction:.2f} vehicles/hour"
)

