import numpy as np


#===================#
# Creating a vector #
#===================#

# A vector of ints
an_integer_vector = np.array([1, 2, 3])
an_integer_vector

an_integer_vector.dtype

# A vector of floats
a_float_vector = np.array([1.7, 2, 3.14])
a_float_vector

a_float_vector.dtype

# A vector of booleans
a_boolean_vector = np.array([True, False, True])
a_boolean_vector

a_boolean_vector.dtype

# A vector of strings
# (Note numpy is entirely happy with unicode
# characters like emojis or Chinese characters!)

a_string_vector = np.array(["Lassie", "盼盼", "Hachi", "Flipper", "🐄"])
a_string_vector

a_string_vector.dtype

# Numbers from 0 to 9
np.arange(10)

# Ones
np.ones(3)

# Zeros
np.zeros(3)

# An array of random values distributed uniformly between 0 and 1
np.random.rand(3)


#===================#
# Numpy Data Types  #
#===================#

type(1)

type(3.14)

as_a_float = np.array([1, 2], dtype="float")
as_a_float
as_a_float.dtype
