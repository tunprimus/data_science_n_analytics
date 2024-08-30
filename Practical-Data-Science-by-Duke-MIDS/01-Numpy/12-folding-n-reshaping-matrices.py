import numpy as np


#================================#
# Folding and Reshaping Matrices #
#================================#

my_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(my_matrix)

my_matrix.reshape((3, 4))

my_matrix.reshape((1, 12))

# This below will generate an error
my_matrix.reshape((3, 3))

#================================#
# Reshaping Dimensions #
#================================#

my_matrix = my_matrix.reshape((2, 2, 3))
print(my_matrix)

my_matrix.shape

#================================#
# Reshape and arange #
#================================#

# One place this reshape trick can be very useful is in working with `np.arange()`. Unlike `ones` and `zeros`, to which one can pass the output dimensions one want, `np.arange()` will always return 1-dimensional data. That means that if one wants a sequence of numbers in a matrix, one has to combine `np.arange()` with .reshape():

np.arange(20)
np.arange(20).reshape((4, 5))

#============#
# Transpose #
#============#

my_matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(my_matrix)

my_matrix.transpose()

# Transpose is often combined with `np.array()` and `.reshape()` because otherwise, those two tools would always generate sequences that increase incrementally across rows before wrapping to the next row:
np.arange(6).reshape((2, 3))

# So if one wants the sequence to move down the columns instead of across the rows, one has to transpose the result:
np.arange(6).reshape((2, 3)).transpose()

# Indeed, .`transpose()` is so frequently used in numpy that you can call it with `.T`:
np.arange(6).reshape((2, 3)).T
