import numpy as np
import time


#================================#
# When Do I Get a View and When Do I Get a Copy? #
#================================#

# NumPy will give a view if one uses simple indexing to get a subset of an array
my_array = np.array([1, 2, 3])
my_subset = my_array[1:3]
my_subset

my_subset[0] = -1
my_array

# But if one asks for a subset any other way—such as with “fancy indexing” (where one passes a list when making the slice) or Boolean subsetting—one will NOT get a view, one will get a copy.
my_array = np.array([1, 2, 3])
my_subset = my_array[[1, 2]]
my_subset[0] = -1
my_array

my_array = np.array([1, 2, 3])
my_slice = my_array[my_array >= 2]
my_slice[0] = -1
my_array


#================================#
# Views and Copies When Editing #
#================================#

# But it is also important to understand what types of modifications of a view will result in changes that propagate back to the original array.
my_array = np.array([1, 2, 3])
my_subset = my_array[1:3]
my_subset = my_subset * 2
my_subset

# We know that because it only used simple indexing, the command my_array[1:3] returned a view, and so my_subset begins its life as a view, sharing data with my_array.
# But is that still true after we run the code my_subset = my_subset * 2?
# The answer is no. That’s because when Python runs the code on the right side of the assignment operator (my_subset * 2) it generates a new array. Then, when it evaluates the assignment operation (my_subset =), it interprets that as a request to reassign the variable my_subset to point at that new array. As a result, by the end of this code, my_subset is no longer a view because the variable my_subset is now associated with a new array.
# As a result, while the values associated with my_subset are 2 times the values that had been there previously, that doubling is not propagating back to my_array:
my_array


# So what would we do if we wanted to keep my_subset a view, and we wanted the doubling of the values in that view to propagate back to my_array?
# In that case, we have to make clear to Python that we aren’t trying to reassign the variable my_subset to a new array, but instead, we are trying to insert the new values created by my_subset * 2 into the original array (the view) associated with my_subset. And we do that by putting [:] on the left-hand side of the assignment operator:

my_array = np.array([1, 2, 3])
my_subset = my_array[1:3]
my_subset[:] = my_subset * 2
my_subset

my_array

"""
Recall from our reading on modifying subsets of vectors that when Python sees an index on the left side of the assignment operator, it interprets that as saying “I’m not trying to reassign the variable `my_subset`, I’m trying to insert these values into the exposed entries of the array associated with `my_subset`.” And if we use `my_subset[:]`, then we’re exposing all the entries in `my_subset` to have new values inserted.

If you have a lot of experience with object-oriented programming (don’t worry if you don’t), then one way to think about this is that using `[]` on the left side of the assignment operator is a lot like assigning a value to a property of an object. Just as `my_object.my_property = "hello"` modifies the property `my_property` associated with the object `my_object` but doesn’t reassign the variable `my_object` to the string `"hello"`, so too does `my_subset[:] = my_subset * 2` modify the underlying array associated with `my_subset` rather than re-assigning it to a new array.
"""


#================================#
# Making a Copy #
#================================#

# So if one wishes to pull a subset of a vector (or array) that is a full copy and not a view, one can just use the .copy() method:
my_vector = np.array([1, 2, 3, 4])
my_subset = my_vector[1:3].copy()
my_subset

my_subset[0] = -99
my_subset

my_vector


#================================#
# How Much Faster Are Views? #
#================================#

# Generate 1 million observations of
# fake efficiency data
efficiency_data = np.random.normal(100, 50, 1_000_000)

degradation_over_time = np.mean(efficiency_data[700_000:1_000_001]) - np.mean(efficiency_data[0:300_000])


start = time.time()

# Let us do the subset 10,000 times and divide
# the overall time taken by 100
# so any small fluctuations in speed average out

# First with simple indexing to get views
for i in range(10_000):
    initial_data = efficiency_data[0:300_000]
    final_data = efficiency_data[700_000:1_000_001]

stop = time.time()
duration_with_views = (stop - start) / 10_000
print(f"Subsets with views took {duration_with_views * 1_000:.4f} milliseconds")

# Fancy indexing *includes* the last endpoint
# so shifted down by 1 from simple indexing
first_subset = np.arange(0, 299_999)
second_subset = np.arange(700_000, 1_000_000)

start = time.time()

# Now do the subset using fancy indexing
# to ensure that we get copies

for i in range(10_000):
    initial_data = efficiency_data[first_subset]
    final_data = efficiency_data[second_subset]

stop = time.time()
duration_with_copies = (stop - start) / 10_000
print(f"Subsets with copies took {duration_with_copies * 1_000:.4f} milliseconds")

print(
    f"Subsets with copies took {duration_with_copies / duration_with_views:,.0f} "
    "times as long as with views"
)
