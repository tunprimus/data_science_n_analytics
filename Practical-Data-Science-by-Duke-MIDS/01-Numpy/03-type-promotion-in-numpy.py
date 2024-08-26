import numpy as np


# If one try pass a list with different kinds of data to `np.array`, numpy will try and be clever and find a way to put all that data in one array by doing something called “Type Promotion.”

np.array(["Nick", 42])

"""
Indeed, there’s a hierarchy of data types, where a type lower on the hierarchy can always be converted into something higher in the order, but not the other way around. That hierarchy is:

Boolean –> integer –> float –> string

When Python is asked to combine data of different types, it will try to move things up this hierarchy by the smallest amount possible in order to make everything the same type.
"""

# For example, if one combine Boolean and float vectors, Python will convert all of the data into float (Remember from our previous reading that Python thinks of True as being like 1, and False as being like 0).

np.array([1, 2.4, True])

# But it doesn’t convert that data into a string vector (even though it could!) because it’s trying to make the smallest movements up that hierarchy that it can.

# But if we try to combine Boolean, float, and string data, Python would be forced to convert everything into a string vector:
np.array([True, 42, "Julio"])
