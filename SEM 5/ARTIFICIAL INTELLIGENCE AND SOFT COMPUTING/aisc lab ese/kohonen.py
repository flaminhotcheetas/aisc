import pandas as pd
import numpy as np

x = [0.2, 0.4]
alpha = 0.2

weights = [
    [0.3, 0.2, 0.1, 0.8, 0.4],
    [0.5, 0.6, 0.7, 0.9, 0.2]
]

weight_df = pd.DataFrame(
    weights,
    index=["x1", "x2"],
    columns=["Y1", "Y2", "Y3", "Y4", "Y5"]
)

print("\nInitial Weight Matrix")
print(weight_df)

distance_rows = []
distances = []

for j in range (len(weights[0])):
  comp1 = (weights[0][j] - x[0])**2
  comp2 = (weights[1][j] - x[1])**2

  distance = comp1 + comp2

  distances.append(distance)

  distance_rows.append({
        "Unit": f"{j+1}",
        "(w1j-x1)^2": round(comp1, 4),
        "(w2j-x2)^2": round(comp2, 4),
        "D(j)": round(distance, 4)
    })

distance_df = pd.DataFrame(distance_rows)

print("\nDISTANCE TABLE")
print(distance_df.to_string(index=False))

winner = distances.index(min(distances))

print("\nWINNER is UNIT", winner+1)
print("Minimum Distance =", round(distances[winner], 4))

update_rows = []

for i in range(len(x)):

    old_weight = weights[i][winner]

    change = alpha * (x[i] - old_weight)

    new_weight = old_weight + change

    update_rows.append({
        "Weight": f"w{i+1}{winner+1}",
        "Old Weight": round(old_weight, 4),
        "Input": x[i],
        "New Weight": round(new_weight, 4)
    })

    weights[i][winner] = new_weight

update_df = pd.DataFrame(update_rows)

updated_weight_df = pd.DataFrame(
    weights,
    index=["x1", "x2"],
    columns=["Y1", "Y2", "Y3", "Y4", "Y5"]
)

print("\nUPDATED WEIGHT MATRIX")
print(updated_weight_df)


