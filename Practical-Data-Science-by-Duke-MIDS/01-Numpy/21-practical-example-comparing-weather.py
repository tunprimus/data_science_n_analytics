import numpy as np


#================================#
# Practical Example: comparing weather by counting like-values #
#================================#

# A matrix containing weather data over a 2-week period for 5 cities. Each entry in the matrix is an integer between 0 and 4 where these represent the following weather conditions:
weather_condition_label = {0: "cloudcover", 1: "sun", 2: "rain", 3: "wind", 4: "snow"}

weather = np.array([[0, 1, 0, 1, 1, 1, 2, 3, 2, 1, 1, 1, 2, 1], [1, 0, 0, 2, 2, 2, 0, 0, 1, 0, 2, 3, 0, 0], [0, 0, 2, 1, 3, 1, 0, 2, 1, 2, 1, 3, 3, 0], [4, 0, 0, 4, 2, 0, 1, 0, 0, 1, 2, 4, 4, 1], [2, 3, 1, 1, 2, 3, 4, 2, 2, 4, 2, 3, 0, 3],])
print(weather)

def count_values(arr):
    # Create an array of 5 zeros, one for each value one wants to count
    counts = np.zeros(5)

    # For each value in the array, increment the count of the value corresponding to that number
    for value in arr:
        counts[value] += 1
    
    return counts

test_input = [2, 1, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
count_values(test_input)

def count_values_in_each_row(matrix):
    # For each row, count the values and output that as a matrix.
    # start with an empty array that we collect our histograms in
    row_counts = []

    for row in matrix:
        counts = count_values(row)
        # Add the result to the output
        row_counts.append(counts)

    return row_counts

count_values_in_each_row(weather)
print(count_values_in_each_row(weather))
