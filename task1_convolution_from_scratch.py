
import numpy as np

# Define the 5x5 input matrix
input_matrix = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
])

# Define the 3x3 filter
kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

stride = 1
padding = 0

# Calculate the output dimensions
input_height, input_width = input_matrix.shape
kernel_height, kernel_width = kernel.shape

output_height = (input_height - kernel_height + 2 * padding) // stride + 1
output_width = (input_width - kernel_width + 2 * padding) // stride + 1

# Create an empty output feature map
output = np.zeros((output_height, output_width), dtype=int)

# Perform convolution manually by sliding the filter
for i in range(output_height):
    for j in range(output_width):

        # Extract the current 3x3 region
        region = input_matrix[
            i * stride:i * stride + kernel_height,
            j * stride:j * stride + kernel_width
        ]

        # Multiply element by element and add the values
        output[i, j] = np.sum(region * kernel)

# Display the results
print("Input Matrix:")
print(input_matrix)

print("\nFilter:")
print(kernel)

print("\nStride:", stride)
print("Padding:", padding)

print("\nOutput Feature Map:")
print(output)

print("\nOutput Shape:", output.shape)

print("\nEffect of changing stride from 1 to 2:")
print("The filter moves two positions at a time instead of one.")
print("The output becomes smaller, changing from 3x3 to 2x2.")
