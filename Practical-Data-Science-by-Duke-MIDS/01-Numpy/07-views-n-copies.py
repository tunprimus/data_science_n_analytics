import numpy as np


#================================#
# Views and Copies #
#================================#

a = np.array([42, 47, -1])
a

new = a[1]
new

my_vector = np.array([1, 2, 3, 4])
my_vector

my_subset = my_vector[1:3]
my_subset

# Now suppose one changes the first entry of my_subset to -99:
my_subset[0] = -99

print(my_vector)

my_vector[2] = 42
print(my_subset)
