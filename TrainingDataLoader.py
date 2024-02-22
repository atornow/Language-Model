import os
import numpy as np


def unique_chars_in_file(file_path):
    # Initialize an empty set to store unique characters
    unique_chars = set()

    # Open the file and read through each character
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Add each character to the set
            unique_chars.update(line)

    # Convert the set of unique characters to a numpy array
    unique_chars_array = np.array(list(unique_chars))

    return unique_chars_array


def create_char_to_binary_map(file_path):
    # Get the unique characters as a numpy array
    unique_chars = unique_chars_in_file(file_path)

    # Initialize a dictionary to hold the mapping
    char_to_binary = {}

    # The length of the binary array will be equal to the number of unique characters
    num_unique_chars = len(unique_chars)

    # Create the mapping
    for index, char in enumerate(unique_chars):
        # Initialize a binary array of zeros
        binary_array = np.zeros(num_unique_chars, dtype=int)
        # Set the value at the current character's index to 1
        binary_array[index] = 1
        # Map the character to its binary array representation
        char_to_binary[char] = binary_array

    return char_to_binary


def file_to_binary_matrix(file_path):
    char_to_binary_map = create_char_to_binary_map(file_path)

    # Read the entire content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Determine the number of unique characters (length of any binary array)
    num_unique_chars = len(next(iter(char_to_binary_map.values())))

    # Initialize an empty list to hold the binary arrays for each character
    binary_matrix = []

    # Convert each character in the file to its binary array representation
    for char in content:
        binary_array = char_to_binary_map.get(char)
        if binary_array is not None:
            binary_matrix.append(binary_array)
        else:
            # If the character is not found in the map, use a zero array
            binary_matrix.append(np.zeros(num_unique_chars, dtype=int))

    # Convert the list of binary arrays into a 2D numpy array (matrix)
    binary_matrix = np.array(binary_matrix)

    return binary_matrix


# Example usage
# First, generate the character to binary mapping
  # Replace with your file path

# Then, create the binary matrix for the file
