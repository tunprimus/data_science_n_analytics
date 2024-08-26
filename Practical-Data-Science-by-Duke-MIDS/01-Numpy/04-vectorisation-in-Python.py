import numpy as np
import time


# Create our data
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])

# Let us first complete this using a non-vectorised approach that uses a for loop…
c = []
for i in range(len(a)):
    c.append(a[i] + b[i])

print("Nonvectorised approach ->", c)

# Now, let us try a vectorised approach
c = a + b

print("Vectorised approach ->", c)

#================================#
# Performance benefits of vectorisation #
#================================#

# Non-vectorised function
def double_nonvectorised(arr):
    doubled = arr.copy()
    for i in range(len(arr)):
        doubled[i] = arr[i] * 2
    return doubled

# Vectorised function
def double_vectorised(arr):
    return arr * 2

array = np.array([1, 2, 3, 4])
print("Nonvectorised = ", double_nonvectorised(array))
print("Vectorised    = ", double_vectorised(array))


# The `time` package has a function `time` that returns the current time in seconds; so taking the difference between `time.time()` between two points in time provides the number of seconds that have elapsed.

def timer(function, argument, num_runs):
    total_time = 0
    # Rerun the code num_runs times
    for _ in range(num_runs):
        t0 = time.time()  # Capture the initial time
        function(argument)  # Run the function we're timing
        t1 = time.time()  # Capture the final time
        total_time += t1 - t0
    return total_time / num_runs  # Return the average across the runs

# Create a very large array to double and time how long it takes
big_array = np.arange(1000000)
num_runs = 5

time_nonvectorised = timer(double_nonvectorised, big_array, num_runs)
time_vectorised = timer(double_vectorised, big_array, num_runs)

print("Nonvectorised code took ", time_nonvectorised, "s")
print("Vectorised code took    ", time_vectorised, "s")
print("Vectorized code was ", time_nonvectorised / time_vectorised, " times faster")

