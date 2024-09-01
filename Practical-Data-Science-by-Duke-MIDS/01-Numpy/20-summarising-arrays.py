import numpy as np


#================================#
# Summarising Arrays #
#================================#

x = np.array([0, 1, 0, -2, 12])

# Each of the summarisation methods shrinks the data down to a single number. A data science term for this is dimensionality reduction
print(f"Sum    = {np.sum(x)}")
print(f"Mean   = {np.mean(x)}")
print(f"Min    = {np.min(x)}")
print(f"Max    = {np.max(x)}")
print(f"Median = {np.median(x)}")
print(f"Size   = {np.size(x)}")
print(f"Count nonzero      = {np.count_nonzero(x)}")
print(f"Standard Deviation = {np.std(x)}")

# For 2D arrays
y = np.array([[0, 1], [2, 3]])
print(y)

print("Sum    =", np.sum(y))
print("Mean   =", np.mean(y))
print("Min    =", np.min(y))
print("Max    =", np.max(y))
print("Median =", np.median(y))
print("Size   =", np.size(y))
print("Count nonzero      =", np.count_nonzero(y))
print("Standard Deviation =", np.std(y))

# To summarise along a specific row or column, use the axis parameter

# Sum of all elements
print(np.sum(y))

# Sum of each column
print(np.sum(y, axis=0))

# Sum of each row
print(np.sum(y, axis=1))

