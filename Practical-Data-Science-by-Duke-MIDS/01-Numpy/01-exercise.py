import numpy as np


# Ques_1. Create a vector with all the prime numbers between 0 and 10 (e.g., just type the prime numbers in a vector).
early_prime_number = np.array([2, 3, 5, 7])
print(early_prime_number)

# Ques_2. Use len() to get the number of numbers you put into your vector.
print(len(early_prime_number))

# Ques_3. Access the `.size` attribute to get the same number (just a different way!)
print(early_prime_number.size)

# Ques_4. What do you think is the `dtype` of vector? Answer without running any code.
print("int")

# Ques_5. Now access the `.dtype` attribute – were you correct?
print(early_prime_number.dtype)
