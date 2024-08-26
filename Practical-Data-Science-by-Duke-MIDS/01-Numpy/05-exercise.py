import numpy as np


# Ques_1. Create a new vector with all the primes between 0 and 20: 2, 3, 5, 7, 11, 13, 17, 19
some_early_prime_nums = np.array([2, 3, 5, 7, 11, 13, 17, 19])
print(some_early_prime_nums)

# Ques_2. Using the `:` operator, subset all the entries from your vector greater than 10.
prime_more_than_10 = some_early_prime_nums[4:]
print(prime_more_than_10)

# Ques_3. Using a logical test (e.g. >, <, ==, !=), subset all the entries from your vector greater than 10.
greater_than_10 = some_early_prime_nums > 10
print(some_early_prime_nums[greater_than_10])

# Ques_4. Using a logical test, subset all the even prime numbers from your list (recall that the `%` operator returns the remainder left over after integer division – for example, `7 % 3` returns 1, since 3 goes into 7 two times with 1 left over).
even_prime = some_early_prime_nums % 2 == 0
print(some_early_prime_nums[even_prime])

# Ques_5. Now, using logical tests, subset all the entries that are either even or greater than 10.
combo = even_prime | greater_than_10
print(some_early_prime_nums[combo])
