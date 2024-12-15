import numpy as np
import re

#from python_codes.pure_maths.test import output_vector


def max_pooling(input_array, pool_size=(2, 2), stride=(2, 2)):

    H, W, C = input_array.shape
    pool_height, pool_width = pool_size
    stride_height, stride_width = stride

    # Calculate output dimensions
    out_height = (H - pool_height) // stride_height + 1
    out_width = (W - pool_width) // stride_width + 1

    # Initialize the output array
    output = np.zeros((out_height, out_width, C))

    for c in range(C):
        for i in range(0, out_height * stride_height, stride_height):
            for j in range(0, out_width * stride_width, stride_width):
                # Apply max pooling for each channel
                output[i // stride_height, j // stride_width, c] = np.max(
                    input_array[i:i+pool_height, j:j+pool_width, c]
                )

    return output

def softmax(x):
    # Subtracting the max value for numerical stability
    x_exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x_exp / np.sum(x_exp, axis=-1, keepdims=True)

import re


# Function to extract float values from the input (ignoring float32, float64, etc.)
def extract_floats_from_file(input_file , output_file):
    with open(input_file ,'r') as f:
        content = f.read()

        # Use regex to find all numbers, ignoring types like float32, float64, etc.
        float_values = re.findall(r"[-+]?\d*\.\d+|\d+", content)

        # Convert extracted strings to float
        extracted_floats = [float(val) for val in float_values]

    # Save the extracted floats to a new file
    with open(output_file, 'w') as f:
        for val in extracted_floats:
            f.write(f"{val}\n")

    print(f"Extracted {len(extracted_floats)} float values and saved to {output_file}")
    return extracted_floats

# Example usage:
# Function to read numbers from a text file and process them
def process_numbers_from_file(file_path):
    with open(file_path, 'r') as file:
        # Read the file, strip any extra spaces/newlines, and convert it into a list of floats
        numbers = [float(line.strip()) for line in file.readlines()]

    # Remove all occurrences of 32
    numbers = [num for num in numbers if num != 32.0]

    processed_numbers = []

    # Loop through the list and apply scaling based on the exponents
    for i in range(0, len(numbers), 2):  # Process the numbers in pairs (value, exponent)
        current_value = numbers[i]
        exponent_value = numbers[i + 1]

        # Apply scaling only if current_value is greater than 1
        if current_value > 1:
            processed_value = current_value * 10 ** (-exponent_value)  # Apply scaling by exponent
        else:
            processed_value = current_value  # Do not scale if the value is <= 1

        # Append the processed value to the list
        processed_numbers.append(processed_value)

    return processed_numbers

def process_and_update_file(file_path):
    # Read the file and process numbers
    processed_numbers = []
    unprocessed_numbers = []

    with open(file_path, 'r') as file:
        # Read all lines in the file
        lines = file.readlines()

        # Process the first 8192 numbers (value, exponent pairs)
        for i in range(0, min(8192, len(lines)), 2):  # Process the numbers in pairs (value, exponent)
            current_value = float(lines[i].strip())
            exponent_value = float(lines[i + 1].strip())

            # Apply scaling by 10 raised to the negative exponent value
            processed_value = current_value * 10 ** (-exponent_value)

            # Append the processed value to the list
            processed_numbers.append(processed_value)

        # Start processing after the first 8192 numbers
        i = 8192

        # Ignore the first 32.0 (skip it)
        if float(lines[i].strip()) == 32.0:
            i += 1  # Skip the 32.0

        # Read the next 64 numbers (biases) as they are
        biases = lines[i:i + 64]
        unprocessed_numbers.extend(biases)
        i += 64  # Skip over the 64 biases

        # Ignore the second 32.0
        if float(lines[i].strip()) == 32.0:
            i += 1  # Skip the second 32.0

        # Process the remaining numbers as they are
        for line in lines[i:]:
            processed_numbers.append(float(line.strip()))

    # Write the processed and unprocessed numbers back into the file
    with open(file_path, 'w') as file:
        # Write the processed numbers
        for num in processed_numbers:
            file.write(f"{num}\n")

        # Write the unprocessed (biases) numbers
        for line in unprocessed_numbers:
            file.write(line)

def process_layer5_file(input_file , output_file,input_vector):
    # extracted_data = extract_floats_from_file(input_file, output_file)
    # processed_values = process_numbers_from_file(output_file)
    # process_and_update_file(output_file)
    with open(output_file, 'r') as f:
        content = f.readlines()

# Extract only float values from the file
        data = [float(value.strip()) for value in content if value.strip()]

        # Check if the data length is sufficient
        if len(data) != 64 * 64 + 64:
            raise ValueError(f"Expected {64*64+64} values, but found {len(data)} in the file.")

        # Reshape data into weights and biases
        weights = np.array(data[:64*64]).reshape((64, 64))  # First 64*64 values as a matrix
        biases = np.array(data[64*64:])  # Last 64 values as a vector

        #input_vector = np.random.rand(64)
        output_vector = np.dot(input_vector, weights) + biases

        print("Input Vector (64):")
        print(input_vector)
        print("\nOutput Vector (64):")
        print(output_vector)
    return output_vector

def process_layer_6_file(file_path,input_data):
    with open(file_path, 'r') as file:
        data = file.read().splitlines()

        # Convert the data into a list of floats
        data = [float(x) for x in data]

        # Extract the first 640 values as weights, reshape them into (64, 10)
        weights = np.array(data[:640]).reshape((64, 10))

        # Extract the next 10 values as biases
        biases = np.array(data[640:650])
        input_data = np.array(input_data)

    # Perform the matrix multiplication between input data and weights
    output = np.dot(input_data, weights) + biases




    # Example usage
    #input_data = np.random.rand(64)  # Example input data with 64 elements

    # Load weights and biases from a text file
    #weights, biases = load_weights_and_biases('layer_6_flat.txt')

    # Perform the dense layer computation
    #output = dense_layer(input_data, weights, biases)

    # Print the output
    print("Output of dense layer:", output)
    return output

