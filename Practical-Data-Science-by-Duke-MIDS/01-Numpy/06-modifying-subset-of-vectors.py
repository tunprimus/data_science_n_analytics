import numpy as np


#================================#
# Modifying Subsets of Vectors #
#================================#

# Create a vector with salaries
salaries = np.array([105_000, 50_000, 55_000, 80_000, 70_000])
salaries

# To modify the 4th entry, one can using
# indexing *on the left side of the assignment operator*
# to assign a new value to only that entry of the array.

salaries[3] = 90_000
salaries

# With logical tests also
salaries[salaries < 60_000] = 65_000
salaries

# If one wanted to give everyone a raise, of course, one could just do:
salaries = salaries + 10_000

# But then one would be giving the raise to everyone, including the person making 75,000 dollars.

# First, re-set our salaries.
salaries = np.array([105_000, 50_000, 55_000, 80_000, 70_000])

# Get the subset of entries that have the lower salaries
lower_salaries = salaries[salaries < 75_000]
lower_salaries

# Increase those lower salaries by ten thousand
new_salaries = lower_salaries + 10_000
new_salaries

# And now the magic: re-insert them
# into the entries of the original
# array that have values less than 75,000
# by putting the subset on the left-hand
# side of the assignment operator.

salaries[salaries < 75_000] = new_salaries
salaries

"""
Note that this last operation worked because there were three entries identified by the logical test on the left-hand side of the assignment operator (`salaries[salaries < 75_000]`), and the array we were assigning to those values (`new_salaries`) also had exactly three entries. As a result, numpy could easily match the entries being assigned to the entries on the left, putting the first assigned value in the first selected entry of `salaries`, the second assigned value in the second selected entry, and the third assigned value into the third selected entry.
"""

# The other thing to note is that while one can do this kind of manipulation in several distinct steps—creating lower_salaries, modifying it to make new_salaries, and then assigning those back into the original salaries—one can also combine all those steps into one line:

# Re-create her original salary vector
salaries = np.array([105_000, 50_000, 55_000, 80_000, 70_000])
salaries

# NumPy can also assign a scalar to multiple elements
my_array = np.array([0, 1, 2, 3])
my_array

# Select middle to entries
my_array[1:3] = -99
my_array

# Select middle to entries.
# Assign vector of length 2 to
# these two entries
my_array[1:3] = np.array([100, 200])
my_array

# But if one try to assign anything that is not either a scalar (single number) or an array of the same size, one will get an error:
# Assign an array of length three
# to two spots in `my_array`:
my_array[1:3] = np.array([100, 200, 300])


#================================#
# Modifying Vectors and Data Types #
#================================#

# NumPy consideration and type promotion only at array creation
np.array([True, False, 7])

# But once a vector has been created, NumPy stops being so considerate: if one tries and cram data of a different type into a vector of a given type, it will try to coerce the data into the established type of the array.
bool_vector = np.array([True, False])
bool_vector

bool_vector[1] = 7
bool_vector


int_vector = np.array([1, 2, 3])
int_vector

int_vector[0] = 42.989723798729874
int_vector

# If one knows one might later need to put a floating point number into int_vector – one could instead tell numpy to make it a floating point number vector at creation:
no_longer_an_int_vector = np.array([1, 2, 3], dtype="float")
no_longer_an_int_vector[0] = 42.989723798729874
no_longer_an_int_vector
