# read csv
import csv

with open("dataset.csv", mode="r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)  # Output: ['value1', 'value2', 'value3']

# write a new csv

import csv

headers = ["Feature1", "Feature2", "Label"]
rows = [
    [0.1, 0.9, 1],
    [0.8, 0.2, 0]
]

with open("output.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)   # Writes a single 1D list (header)
    writer.writerows(rows)     # Writes a 2D list (multiple rows)


# append in a csv

import csv

new_log = [0.45, 0.55, 1]

with open("output.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(new_log)  # Appends to the bottom of the file

#reading csv into a dict

import csv

with open("dataset.csv", mode="r") as file:
    # Automatically treats the first row as dictionary keys
    reader = csv.DictReader(file) 
    for row in reader:
        print(row["Feature1"], row["Label"])


#writing csv from a dict

import csv

fields = ["Name", "Score"]
data = [
    {"Name": "Alice", "Score": 90},
    {"Name": "Bob", "Score": 85}
]

with open("scores.csv", mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()  # Writes the fieldnames as the top row
    writer.writerows(data)

#filter rows based on condn

import csv

with open("dataset.csv", mode="r") as file_in, open("filtered.csv", mode="w", newline="") as file_out:
    reader = csv.reader(file_in)
    writer = csv.writer(file_out)
    
    writer.writerow(next(reader)) # Copy the header directly to the new file
    
    for row in reader:
        if int(row[2]) == 1: # Checking the target/label column
            writer.writerow(row)


#compute dataset - mean/avg

import csv

total_loss = 0
count = 0

with open("training_log.csv", mode="r") as file:
    reader = csv.reader(file)
    next(reader) # Skip header
    
    for row in reader:
        total_loss += float(row[1]) # Assume Loss is in the second column
        count += 1

mean_loss = total_loss / count
print(f"Dataset Size: {count} rows. Mean Loss: {mean_loss:.4f}")


#if numpy allowed and installed
import numpy as np

# Loads numerical data directly into a matrix, skipping the text header
matrix = np.loadtxt("dataset.csv", delimiter=",", skiprows=1)
print(matrix.shape)

