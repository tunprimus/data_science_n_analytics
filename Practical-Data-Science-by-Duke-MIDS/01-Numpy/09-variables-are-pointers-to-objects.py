import numpy as np


#================================#
# Variables are pointers to objects #
#================================#

my_array = np.array([1,-12,42])
id(my_array)

# A view points to the same memory address
my_view = my_array
id(my_view)

# A copy will point to a different place in memory than did our original variable
my_copy = my_array.copy()
id(my_copy)


"""
Review of Views and Copies

The last reading on views and copies covered a lot, so here’s a brief summary of the key takeaways:

    - When an array is subset using simple indexing (i.e., by passing an index or range of indices denoted with a :), the result is just a reference to the original array’s data. This is called a view.

    - Because the view created through simple indexing is sharing data with the original array, changes to one will also impact the other.

    - While we often only refer to the newly created array as a “view,” the relationship between the original array and the view is symmetric, meaning changes to either may impact the other (if the change impacts an entry that is shared).

    - When you create a subset using fancy indexing or Boolean subsetting with a logical test, numpy will create a copy, not a view.

    - A view can be converted into a copy with .copy().

    - Creating a view is much faster than creating a copy; with that said, for most sizes of datasets you will encounter in life, both are exceedingly fast.


"""
