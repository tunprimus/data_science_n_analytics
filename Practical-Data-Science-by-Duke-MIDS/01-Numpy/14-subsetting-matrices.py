import numpy as np

#================================#
# Subsetting by Simple Index #
#================================#

our_matrix = np.arange(12).reshape((3, 4))
print(our_matrix)

# Get entry from 2nd row and 3rd column
print(our_matrix[1, 2])

# Get the four elements in the top left corner of the matrix
print(our_matrix[0:2, 0:2])

# The one behaviour that comes up in matrices that tends not to come up in vectors is that if one wants ALL entries along a specific dimension, one still puts in a comma and type : for the dimension on which one wants all observations. So get all the columns in the second row (index 1), type:

print(our_matrix[1, :])

# Or if one wants all the rows of the third column, type:
print(our_matrix[:, 2])


#================================#
# Subsetting with Logicals #
#================================#

survey = np.array([[20, 22_000, 12], [35, 65_000, 16], [55, 19_000, 11], [45, 35_000, 12]])
print(survey)

# If one wants to select all the rows where income was less than the US median income (about 64,000), one would first extract the income column, then create a logical column that’s TRUE if income is below 65,000, then put that in the first position of tbe square brackets:
income = survey[:, 1]
print(income)

below_median = income < 64000
print(below_median)

survey[below_median, :]

# Or, of course, one could do that all in one line instead of breaking out the steps:

survey[survey[:, 1] < 64000, :]


#================================#
# Subsetting by Row and Column Simultaneously #
#================================#

# Suppose one wants the education levels of everyone with incomes below the US median. One could do this in two steps by subsetting rows and then subsetting columns
below_median = survey[survey[:, 1] < 64000, :]
below_median[:, 2]

# Or one can do it all in one command!
survey[survey[:, 1] < 64000, 2]

# So what is the average education of people earning less than the median income in the US in the toy data?
np.mean(survey[survey[:, 1] < 64000, 2])


#================================#
# Fancy Indexing #
#================================#

# Basically, to do fancy indexing one passes two lists or arrays separated by a comma, then the entries of those lists are paired up to create coordinates. So for example, the following code:
our_matrix[[0, 1, 2], [0, 1, 2]]

# Is pulling the entry at 0, 0, the entry at 1, 1, and the entry at 2, 2 (e.g. the matrix’s diagonal).
