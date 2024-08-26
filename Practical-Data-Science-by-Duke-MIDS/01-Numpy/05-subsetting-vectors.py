import numpy as np


#================================#
# What is Subsetting? #
#================================#

a = np.array([42, 47, -1])
a

a[1]

# This `variable[]` notation is something we’ll use a lot with numpy, and it will always mean the same thing: we’re trying to access some data in the data stored in the box `variable`.
new = a[1]
new

#================================#
# Subsetting By Index #
#================================#

a[0:2]


#================#
# Fancy Indexing #
#================#

a[[0, 2]]

zero_and_two = np.array([0, 2])
zero_and_two

a[zero_and_two]

# Note that with fancy indexing, you do not have to subset entries in order: if you pass indices out of order, you will get a vector with a new order!

a[[2, 0]]


#================#
# Subsetting with Booleans #
#================#

# Within these brackets is a vector with the same number of Boolean elements as there are elements in the vector you want to subset. Elements across the two vectors are matched by order: elements that match with `True` are kept while elements that match with `False` are dropped.
fruits = np.array(["dogs", "cats"])
fruits[[True, False]]

a = np.array([42, 47, -1])
my_subset = np.array([True, False, True])
b = a[my_subset]
b

#================#
# Subsetting With Logical Operations #
#================#
# WARNING: When working with numpy arrays, we can’t use the logical operations `or`, `and`, and `not` we use in vanilla Python. Instead, when working True and False with numpy vectors, we have to use `&` for “and”, `|` for “or”, and `~` for “not”.

# Create a numeric vector
# Reminder: if one pass a third argument to `np.arange()`,
# numpy uses that to determine the step sizes between values!

numbers = np.arange(10, 110, 10)
numbers

# Get small numbers:
numbers[numbers <= 50]

# Combine logical conditions
numbers[(numbers < 30) | (numbers == 100)]

# This gives an error because no parenthesis is used
# numbers[numbers < 30 | numbers == 100]

# Get only the middle set of numbers
middle_number = (30 < numbers) & (numbers < 80)
middle_number

numbers[middle_number]
