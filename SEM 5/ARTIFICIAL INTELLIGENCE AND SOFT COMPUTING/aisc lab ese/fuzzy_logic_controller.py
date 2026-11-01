# Fuzzy Controller for Temperature Regulation in a Room

# Step 1: Install and import required libraries
# pip -q install scikit-fuzzy

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd


# Step 2: Define input and output variable
temperature = ctrl.Antecedent(
    np.arange(0, 51, 1), 'temperature'
)
humidity = ctrl.Antecedent(
    np.arange(0, 101, 1), 'humidity'
)
cooling_power = ctrl.Consequent(
    np.arange(0, 101, 1), 'cooling_power'
)

# Step 3: Define membership functions for temperature
temperature['too_cold'] = fuzz.trimf(temperature.universe, [0, 0, 10])
temperature['cold'] = fuzz.trimf(temperature.universe, [5, 12, 20])
temperature['comfortable'] = fuzz.trimf(temperature.universe, [17, 22, 27])
temperature['warm'] = fuzz.trimf(temperature.universe, [24, 30, 36])
temperature['hot'] = fuzz.trimf(temperature.universe, [33, 50, 50])

# Step 4: Define membership functions for humidity
humidity['low'] = fuzz.trimf( humidity.universe, [0, 0, 40])
humidity['medium'] = fuzz.trimf(humidity.universe, [20, 50, 80])
humidity['high'] = fuzz.trimf(humidity.universe, [60, 100, 100])

# Step 5: Define membership functions for cooling power
cooling_power['off'] = fuzz.trimf(cooling_power.universe, [0, 0, 15])
cooling_power['low'] = fuzz.trimf(cooling_power.universe, [0, 25, 50])
cooling_power['medium'] = fuzz.trimf(cooling_power.universe, [25, 50, 75])
cooling_power['high'] = fuzz.trimf(cooling_power.universe, [50, 75, 100])
cooling_power['maximum'] = fuzz.trimf(cooling_power.universe, [85, 100, 100])

# Step 6: Display membership function graphs
temperature.view()
plt.title("Temperature Membership Functions")
plt.show()
humidity.view()
plt.title("Humidity Membership Functions")
plt.show()
cooling_power.view()
plt.title("Cooling Power Membership Functions")
plt.show()

# Step 7: Define the fuzzy rule base
rule1 = ctrl.Rule(temperature['too_cold'] & humidity['low'],cooling_power['off'])

rule2 = ctrl.Rule(temperature['too_cold'] & humidity['medium'],cooling_power['off'])

rule3 = ctrl.Rule(temperature['too_cold'] & humidity['high'],cooling_power['off'])

rule4 = ctrl.Rule(temperature['cold'] & humidity['low'],cooling_power['off'])

rule5 = ctrl.Rule(temperature['cold'] & humidity['medium'],cooling_power['off'])

rule6 = ctrl.Rule(temperature['cold'] & humidity['high'],cooling_power['low'])

rule7 = ctrl.Rule(temperature['comfortable'] & humidity['low'],cooling_power['off'])

rule8 = ctrl.Rule(temperature['comfortable'] & humidity['medium'],cooling_power['off'])

rule9 = ctrl.Rule(temperature['comfortable'] & humidity['high'],cooling_power['low'])

rule10 = ctrl.Rule(temperature['warm'] & humidity['low'],cooling_power['low'])

rule11 = ctrl.Rule(temperature['warm'] & humidity['medium'],cooling_power['medium'])

rule12 = ctrl.Rule(temperature['warm'] & humidity['high'],cooling_power['high'])

rule13 = ctrl.Rule(temperature['hot'] & humidity['low'],cooling_power['medium'])

rule14 = ctrl.Rule(temperature['hot'] & humidity['medium'],cooling_power['high'])

rule15 = ctrl.Rule(temperature['hot'] & humidity['high'],cooling_power['maximum'])


# Step 8: Create the fuzzy control system
room_controller = ctrl.ControlSystem([
    rule1, rule2, rule3,
    rule4, rule5, rule6,
    rule7, rule8, rule9,
    rule10, rule11, rule12,
    rule13, rule14, rule15
])

# Step 9: Take user inputs

print("Fuzzy Controller for Room Temperature Regulation")
try:
    room_temp = float(input("Enter room temperature (0-50 °C): "))
    room_humidity = float(input("Enter room humidity (0-100%): "))
    if not (0 <= room_temp <= 50):
        raise ValueError("Temperature must be between 0 and 50 °C.")
    if not (0 <= room_humidity <= 100):
        raise ValueError("Humidity must be between 0 and 100%.")
except ValueError as error:
    raise ValueError(f"Invalid input: {error}")

# Step 10: Perform fuzzification
print("\nFuzzification")
print("Room Temperature:", room_temp, "°C")
print("Room Humidity:", room_humidity, "%")
temp_too_cold = fuzz.interp_membership(
    temperature.universe,
    temperature['too_cold'].mf,
    room_temp
)
temp_cold = fuzz.interp_membership(
    temperature.universe,
    temperature['cold'].mf,
    room_temp
)

temp_comfortable = fuzz.interp_membership(
    temperature.universe,
    temperature['comfortable'].mf,
    room_temp
)

temp_warm = fuzz.interp_membership(
    temperature.universe,
    temperature['warm'].mf,
    room_temp
)

temp_hot = fuzz.interp_membership(
    temperature.universe,
    temperature['hot'].mf,
    room_temp
)

print("\nTemperature Membership Values")
print("Too Cold:", round(temp_too_cold, 3))
print("Cold:", round(temp_cold, 3))
print("Comfortable:", round(temp_comfortable, 3))
print("Warm:", round(temp_warm, 3))
print("Hot:", round(temp_hot, 3))

humidity_low = fuzz.interp_membership(
    humidity.universe,
    humidity['low'].mf,
    room_humidity
)

humidity_medium = fuzz.interp_membership(
    humidity.universe,
    humidity['medium'].mf,
    room_humidity
)

humidity_high = fuzz.interp_membership(
    humidity.universe,
    humidity['high'].mf,
    room_humidity
)

print("\nHumidity Membership Values")
print("Low:", round(humidity_low, 3))
print("Medium:", round(humidity_medium, 3))
print("High:", round(humidity_high, 3))


# Step 11: Evaluate fuzzy rules

simulation = ctrl.ControlSystemSimulation(room_controller)

simulation.input['temperature'] = room_temp
simulation.input['humidity'] = room_humidity

simulation.compute()

result = simulation.output['cooling_power']


# Step 12: Display defuzzified output

print("\nDefuzzification")
print("Recommended Cooling Power:", round(result, 2), "%")

if result < 15:
    status = "Off"
elif result < 35:
    status = "Low"
elif result < 60:
    status = "Medium"
elif result < 85:
    status = "High"
else:
    status = "Maximum"

print("Cooling Status:", status)


# Step 13: Display the aggregated output and centroid

cooling_power.view(sim=simulation)
plt.title("Defuzzification: Room Cooling Power")
plt.show()


# Step 14: Test the controller with multiple test cases

test_cases = [
    (5, 30),
    (15, 40),
    (22, 50),
    (28, 40),
    (32, 70),
    (38, 60),
    (45, 90),
    (25, 85)
]

results = []

for temp, hum in test_cases:
    sim = ctrl.ControlSystemSimulation(room_controller)

    sim.input['temperature'] = temp
    sim.input['humidity'] = hum

    sim.compute()

    power = sim.output['cooling_power']

    if power < 15:
        cooling_status = "Off"
    elif power < 35:
        cooling_status = "Low"
    elif power < 60:
        cooling_status = "Medium"
    elif power < 85:
        cooling_status = "High"
    else:
        cooling_status = "Maximum"

    results.append({
        'Temperature (°C)': temp,
        'Humidity (%)': hum,
        'Cooling Power (%)': round(power, 2),
        'Cooling Status': cooling_status
    })

results_df = pd.DataFrame(results)

print("\nTest Case Results")
display(results_df)


# Step 15: Plot temperature versus cooling power

plt.figure(figsize=(9, 5))

plt.plot(
    results_df['Temperature (°C)'],
    results_df['Cooling Power (%)'],
    marker='o',
    label='Cooling Power'
)

plt.xlabel("Room Temperature (°C)")
plt.ylabel("Cooling Power (%)")
plt.title("Fuzzy Controller for Room Temperature Regulation")
plt.grid(True)
plt.legend()
plt.show()


# Step 16: Define a reusable room control function

def control_room(temp, hum):
    if not (0 <= temp <= 50):
        raise ValueError(
            "Temperature must be between 0 and 50 °C."
        )

    if not (0 <= hum <= 100):
        raise ValueError(
            "Humidity must be between 0 and 100%."
        )

    sim = ctrl.ControlSystemSimulation(room_controller)

    sim.input['temperature'] = temp
    sim.input['humidity'] = hum

    sim.compute()

    power = sim.output['cooling_power']

    print("\nRoom Temperature Controller")
    print("Room Temperature:", temp, "°C")
    print("Room Humidity:", hum, "%")
    print("Recommended Cooling Power:", round(power, 2), "%")

    if power < 15:
        print("Status: Cooling Off")
    elif power < 35:
        print("Status: Low Cooling")
    elif power < 60:
        print("Status: Medium Cooling")
    elif power < 85:
        print("Status: High Cooling")
    else:
        print("Status: Maximum Cooling")

    return power
