import numpy as np


#===================#
# Math with Scalars #
#===================#

# Suppose we are working with data on car sales, and we have the value of all the cars we sold last year
sales = np.array([34_255, 27_222, 42_250, 12_000])
sales

# Now suppose that for every car we sell,
# we have to pay a sales tax of 10%.
# How could we calculate the after-tax revenue
# from each of the sales?

# Simple!
after_tax = sales * 0.90

# And suppose we also had to pay a
# flat fee of 500 dollars to process each
# sale. Now what would our final revenue be?

final = after_tax - 500
final

# For example, suppose we wanted to identify sales for more than $30,000. We could do:

final > 30_000

# Round to the nearest dollar
np.round(final)


#================================#
# Math with Equal-Length Vectors #
#================================#

# Two vectors with the same number of elements 
numbers = np.arange(5)
numbers

numbers2 = np.array([0, 0, 1, 1, 0])
numbers2

numbers3 = numbers2 + numbers
numbers3

# Suppose that in addition to information about the sale price of all of the cars we sold last year, we also had data on what those cars cost us (the dealership):
prices = np.array([27_750, 23_500, 39_200, 6_700])
prices

# Now we can combine the after-tax revenue we made from each sale with what the cars we sold cost the dealership to estimate the net profit from each sale:

final - prices



#==============#
# Other Shapes #
#==============#

# But if you try an operation with two vectors of different lengths, and one isn’t of size one, you get an error
vec1 = np.array([1, 2, 3])
vec2 = np.array([1, 2, 3, 4, 5, 6])
vec1 + vec2


#================================#
# Vectorised Code / Vectorisation #
#================================#

# Either this or one get lists
vector1 = np.array([1, 2, 3, 4, 5])
vector2 = np.array([6, 7, 8, 9, 10])

results = list()
for i in range(len(vector1)):
    # Note you can pull items from a vector like 
    # items from a list with `[]`.
    summation = vector1[i] + vector2[i]
    results.append(summation)
print(results)

#================================#
# Summarising Vectors #
#================================#

# Toy height vector
# (Obviously you could easily find the tallest, shortest, etc.
# in this data set without numpy -- this is just an example!)
heights_in_cm = np.array([155, 171, 162, 170, 171])

# Tallest
np.max(heights_in_cm)

# Shortest
np.min(heights_in_cm)

# Median
np.median(heights_in_cm)

# Standard deviation
np.std(heights_in_cm)

# Here is a short (very incomplete!) list of these kinds of functions:
# len(numbers) # number of elements 
# np.max(numbers) # maximum value
# np.min(numbers) # minimum value
# np.sum(numbers) # sum of all entries
# np.mean(numbers) # mean
# np.median(numbers) # median
# np.var(numbers) # variance
# np.std(numbers) # standard deviation
# np.quantile(numbers, q) # the qth quintile of numbers

# Suppose we wanted to know the number of sales that generated more than $30,000 in revenue. First, we could do the same manipulation we did up top:
large_sales = final > 30_000
large_sales

# Then we can sum up that vector (remember: True in Python is treated like 1 and False is treated like 0 when passed to functions like np.sum() and np.mean()):

np.sum(large_sales)
