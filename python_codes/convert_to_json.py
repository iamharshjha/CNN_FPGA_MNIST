import json
import os
import ast
import re

# Directory containing .txt files
directory = './weights'

# Function to extract numbers from the string and exclude 32
def extract_numbers_from_string(line):
    # Find all numbers (integers or floats) in the string using regular expression
    numbers = [float(num) for num in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
    # Exclude 32 from the list
    return [num for num in numbers if num != 32]

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        txt_file_path = os.path.join(directory, filename)

        # List to store all extracted numbers
        numbers = []

        with open(txt_file_path, 'r') as file:
            for line in file:
                line = line.strip()  # Remove leading/trailing whitespace
                if line:  # If line is not empty
                    # Extract numbers from the line and exclude 32
                    numbers.extend(extract_numbers_from_string(line))

        # Debugging: Print out the numbers to verify
        print(f"Processed numbers from {filename}: {numbers}")

        # Only save if there are numbers
        if numbers:
            json_filename = f"{os.path.splitext(filename)[0]}.json"
            json_file_path = os.path.join(directory, json_filename)

            with open(json_file_path, 'w') as json_file:
                json.dump(numbers, json_file, indent=4)

            print(f"Converted {filename} to {json_filename}")
        else:
            print(f"No valid data found in {filename}. Skipping conversion.")
