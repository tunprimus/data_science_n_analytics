import numpy as np
import time
from pympler import asizeof
# Make a regular Python list
# with all the numbers up to one hundred million

# Remember `range` doesn't include the last number,
# so I have to go up to 100_000_001 to actually get all
# the numbers from 1 to 100_000_000

ONE_HUNDRED_MIL_LIST = list(range(1, 100_000_001))

# Now make a numpy vector
# with all the numbers up to one hundred million

ONE_HUNDRED_MIL_VECTOR = np.arange(1, 100_000_001)

start_time_list = time.time()

total = 0
for i in ONE_HUNDRED_MIL_LIST:
    total = total + i

end_time_list = time.time()
python_total = end_time_list - start_time_list
print(f"Python took {python_total:.3f} seconds")

# Now for NumPy
start_time_vec = time.time()

# Now we sum up all the numbers in the array
# using the numpy `sum` function.
np.sum(ONE_HUNDRED_MIL_VECTOR)

end_time_vec = time.time()
numpy_total = start_time_vec - end_time_vec
print(f"NumPy took {numpy_total:.3f} seconds")

print(f"NumPy was {python_total / numpy_total:.1f}x faster!")

# `asizeof.asizeof()` gets the size of an object
# and all of its contents in bytes, so we'll 
# divide it's output by one billion to get 
# the value in gigabytes.

list_size_in_gb = asizeof.asizeof(ONE_HUNDRED_MIL_LIST) / 1_000_000_000
vector_size_in_gb = asizeof.asizeof(ONE_HUNDRED_MIL_VECTOR) / 1_000_000_000

print(f"The Python list of 100_000_000 numbers took up {list_size_in_gb:.2f} GB of RAM")
print(f"The numpy vector of 100_000_000 numbers took up {vector_size_in_gb:.2f} GB of RAM")
print(
    f"That means the Python list took up {list_size_in_gb/vector_size_in_gb:.0f}x "
    "as much space as the numpy vector!"
)

print("\n ======= Now for 250 Million Elements ======= \n")

# Now up to 500 million
TWO_FIFTY_HUNDRED_MIL_LIST = list(range(1, 250_000_001))
TWO_FIFTY_HUNDRED_MIL_VECTOR = np.arange(1, 250_000_001)

start_time_list2 = time.time()

total2 = 0
for i in TWO_FIFTY_HUNDRED_MIL_LIST:
    total2 = total + i

end_time_list2 = time.time()
python_total2 = end_time_list2 - start_time_list2
print(f"Python took {python_total2:.3f} seconds for 250 million list")


start_time_vec2 = time.time()
np.sum(TWO_FIFTY_HUNDRED_MIL_VECTOR)

end_time_vec2 = time.time()
numpy_total2 = start_time_vec2 - end_time_vec2
print(f"NumPy took {numpy_total2:.3f} seconds")

print(f"NumPy was {python_total2 / numpy_total2:.1f}x faster for 250 million elements!")

list_size_in_gb2 = asizeof.asizeof(TWO_FIFTY_HUNDRED_MIL_LIST) / 1_000_000_000
vector_size_in_gb2 = asizeof.asizeof(TWO_FIFTY_HUNDRED_MIL_VECTOR) / 1_000_000_000

print(f"The Python list of 250_000_000 numbers took up {list_size_in_gb2:.2f} GB of RAM")
print(f"The numpy vector of 250_000_000 numbers took up {vector_size_in_gb2:.2f} GB of RAM")
print(
    f"That means the Python list took up {list_size_in_gb2/vector_size_in_gb2:.0f}x "
    "as much space as the numpy vector!"
)
"""
Results

Python took 13.462 seconds
NumPy took -0.091 seconds
NumPy was -147.3x faster!
The Python list of 100_000_000 numbers took up 4.00 GB of RAM
The numpy vector of numbers took up 0.80 GB of RAM
That means the Python list took up 5x as much space as the numpy vector!

======= Now for 250 Million Elements ======= 

Python took 28.869 seconds for 250 million list
NumPy took -0.222 seconds
NumPy was -129.9x faster for 250 million elements!

Killed

"""
