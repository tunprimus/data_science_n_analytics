import numpy as np


#================================#
# Broadcasting #
#================================#

# Vector of same length
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(v1 + v2)

# Vector with scalar
print(v1 + 1)

print(v1 + np.array([1]))

# There is error if the vectors are not the same length
v1 = np.array([1, 2, 3])
v2 = np.array([1, 2,])
print(v1 + v2)

#================================#
# Broadcasting Rules #
#================================#

# If arrays are of different dimensions,
# 1a. See if the rightmost dimensions are the same length or 1.
# If so, they are broadcastable.
# 2. Proceed to the next dimension and repeat the process.
# 3. If all dimensions are broadcastable, the arrays are broadcastable.

my_vector = np.array([1, 2, 3])
print(my_vector)
my_matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(my_matrix)

print(my_vector + my_matrix)

my_matrix2 = np.array([[1, 2], [3, 4], [5, 6]])
print(my_vector + my_matrix2)


#================================#
# A Common Gotcha: Narrow Matrices v. 1-Dimensional Vectors #
#================================#

# In numpy, there is a distinction between a 1-dimensional vector and a 2-dimensional matrix with only 1 row or 1 column.
my_vector = np.array([1, 2, 3])
print(my_vector)
print(my_vector.shape)

skinny_matrix = np.array([[1], [2], [3]])
print(skinny_matrix)
print(skinny_matrix.shape)

print(my_vector + my_vector)

# Broadcasting the skinny matrix against the 1D vector
print(skinny_matrix + my_vector)

# To avoid the behaviour, use reshape
now_a_vector = skinny_matrix.reshape(3)
print(now_a_vector)
print(now_a_vector.shape)

print(now_a_vector + my_vector)

# Or one could use .reshape to make the original one-dimensional vector a matrix with the same shape as the skinny_matrix:
now_a_matrix = my_vector.reshape((3, 1))
print(now_a_matrix)
print(now_a_matrix.shape)

print(skinny_matrix + now_a_matrix)
